"""Linkage-quality evaluation: stratified labelling samples, metrics, silver-label checks."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import deterministic as det
from .store import Warehouse

log = logging.getLogger(__name__)

SIDE_COLS = ["event_date_local", "reg_norm", "msn", "make_model_raw", "type_family", "operator_raw",
             "flight_no_norm", "location_raw", "country_iso", "fatalities", "damage", "url"]


def sample_for_labelling(wh: Warehouse, out_csv: str | Path, per_stratum: int = 40,
                         seed: int = 7) -> pd.DataFrame:
    """Stratify by source pair × method × score band; oversample the grey zone.

    Strata: score bands [0.3,0.6), [0.6,0.9), [0.9,0.99), [0.99,1]. The two middle
    bands get 2× quota because that's where errors concentrate.
    """
    links = wh.con.execute("SELECT * FROM record_link WHERE method <> 'SAME_EVENT'").df()
    if links.empty:
        raise ValueError("no links to sample")
    links["pair"] = links["left_source"] + "↔" + links["right_source"]
    links["band"] = pd.cut(links["score"], [0, 0.6, 0.9, 0.99, 1.0001], right=False,
                           labels=["<0.6", "0.6-0.9", "0.9-0.99", ">=0.99"])
    quota = {"<0.6": per_stratum, "0.6-0.9": 2 * per_stratum, "0.9-0.99": 2 * per_stratum, ">=0.99": per_stratum}
    parts = []
    for (pair, method, band), g in links.groupby(["pair", "method", "band"], observed=True):
        parts.append(g.sample(min(len(g), quota[str(band)]), random_state=seed))
    s = pd.concat(parts, ignore_index=True)

    recs = wh.records().set_index(["source", "source_id"])
    L = recs.loc[list(zip(s.left_source, s.left_id)), SIDE_COLS].add_prefix("l_").reset_index(drop=True)
    R = recs.loc[list(zip(s.right_source, s.right_id)), SIDE_COLS].add_prefix("r_").reset_index(drop=True)
    sheet = pd.concat([s[["left_source", "left_id", "right_source", "right_id", "pair", "method",
                          "band", "score", "status"]], L, R], axis=1)
    sheet["is_match"] = ""   # labeller fills 1 / 0
    sheet["note"] = ""
    sheet.to_csv(out_csv, index=False)
    log.info("wrote %d pairs to label → %s", len(sheet), out_csv)
    return sheet


def import_labels(wh: Warehouse, labels_csv: str | Path, labeller: str = "analyst") -> int:
    df = pd.read_csv(labels_csv, dtype=str)
    df = df[df["is_match"].isin(["0", "1", "true", "false", "True", "False", "yes", "no"])]
    df["is_match"] = df["is_match"].str.lower().isin(["1", "true", "yes"])
    df["labeller"], df["labelled_at"] = labeller, pd.Timestamp.now('UTC').tz_localize(None)
    df["note"] = df.get("note", "")
    cols = ["left_source", "left_id", "right_source", "right_id", "is_match", "labeller", "labelled_at", "note"]
    wh.con.register("_lab", df[cols])
    wh.con.execute("""
        DELETE FROM link_label USING _lab
        WHERE link_label.left_source=_lab.left_source AND link_label.left_id=_lab.left_id
          AND link_label.right_source=_lab.right_source AND link_label.right_id=_lab.right_id""")
    wh.con.execute("INSERT INTO link_label SELECT * FROM _lab")
    # Human labels are also final decisions on the link itself.
    wh.con.execute("""
        UPDATE record_link k SET status = CASE WHEN l.is_match THEN 'accepted' ELSE 'rejected' END,
               decided_by = 'manual:' || l.labeller, decided_at = l.labelled_at
        FROM _lab l
        WHERE k.left_source=l.left_source AND k.left_id=l.left_id
          AND k.right_source=l.right_source AND k.right_id=l.right_id""")
    wh.con.unregister("_lab")
    return len(df)


def metrics(wh: Warehouse) -> pd.DataFrame:
    """Precision / recall / F1 of the *machine* decision vs human labels, by source pair and method.

    `predicted` = the machine score clears its auto-accept threshold (scores are never
    overwritten by human labels, so this measures the machine, not the reviewer).
    Recall here is relative to labelled positives among candidates; pair it with
    `blocking_recall_vs_t0` to see what the candidate generator never surfaced.
    """
    df = wh.con.execute("""
        SELECT l.*, k.method, k.score,
               (k.score >= CASE k.method WHEN 'splink' THEN 0.95 ELSE 0.85 END) AS predicted
        FROM link_label l
        LEFT JOIN record_link k USING (left_source, left_id, right_source, right_id)
    """).df()
    if df.empty:
        return df
    df["predicted"] = df["predicted"].fillna(False).astype(bool)
    df["pair"] = df["left_source"] + "↔" + df["right_source"]
    rows = []
    for keys, g in list(df.groupby(["pair", "method"], dropna=False)) + [(("ALL", "ALL"), df)]:
        tp = int((g.predicted & g.is_match).sum())
        fp = int((g.predicted & ~g.is_match).sum())
        fn = int((~g.predicted & g.is_match).sum())
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * p * r / (p + r) if p == p and r == r and (p + r) else float("nan")
        rows.append(dict(pair=keys[0], method=keys[1], n=len(g), tp=tp, fp=fp, fn=fn,
                         precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4)))
    return pd.DataFrame(rows)


def blocking_recall_vs_t0(wh: Warehouse, cfg: det.DetConfig | None = None) -> pd.DataFrame:
    """Of the cross-source T0 (explicit-ID) links, how many would rules T1–T3 have found on their own?

    T0 is a silver standard biased toward well-documented accidents, so treat this as an
    upper bound on rule recall, not the real thing.
    """
    cfg = cfg or det.DetConfig()
    return wh.con.execute(f"""
        WITH t0 AS (SELECT * FROM record_link WHERE method = 'T0' AND left_source <> right_source),
        rules AS ({det.tiers_sql(cfg)})
        SELECT t0.left_source || '↔' || t0.right_source AS pair, count(*) AS t0_links,
               count(rules.method) AS also_found_by_rules,
               round(count(rules.method) / count(*), 4) AS rule_recall_on_t0,
               count(*) FILTER (WHERE rules.method = 'T1') AS via_T1,
               count(*) FILTER (WHERE rules.method = 'T2') AS via_T2,
               count(*) FILTER (WHERE rules.method = 'T3') AS via_T3
        FROM t0 LEFT JOIN rules USING (left_source, left_id, right_source, right_id)
        GROUP BY 1 ORDER BY 1
    """).df()


def review_queue(wh: Warehouse, out_csv: str | Path | None = None) -> pd.DataFrame:
    q = wh.con.execute("""
        SELECT k.*, l.event_date_local AS l_date, l.reg_norm AS l_reg, l.type_family AS l_type,
               r.event_date_local AS r_date, r.reg_norm AS r_reg, r.type_family AS r_type
        FROM record_link k
        JOIN source_record l ON l.source = k.left_source AND l.source_id = k.left_id
        JOIN source_record r ON r.source = k.right_source AND r.source_id = k.right_id
        WHERE k.status = 'needs_review' ORDER BY k.score DESC
    """).df()
    if out_csv:
        q.to_csv(out_csv, index=False)
    return q
