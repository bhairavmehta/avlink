"""Probabilistic layer: Fellegi-Sunter via Splink (DuckDB backend).

Runs across all three sources (link_only), then keeps only pairs that the
deterministic tiers did NOT already decide. Output lands in record_link with
method='splink' and status:

    p >= accept_threshold          → auto_accept
    review_low <= p < accept       → needs_review   (→ LLM adjudicator / humans)
    p <  review_low                → dropped

Training
--------
* u: random sampling (no labels).
* m: if there are ≥ `min_t0_labels` T0 (explicit NTSB-number/docket) links, they're
  used as pairwise labels — T0 is independent of registration/flight fields, so it
  doesn't bias their m-probabilities. Otherwise EM in two sessions with
  complementary blocking (registration-blocked, then date+type-blocked).
* Registration and MSN are strongly correlated; MSN is NOT a separate comparison
  (it's used as a conflict guard in the deterministic layer instead).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import pandas as pd

from .store import Warehouse

log = logging.getLogger(__name__)

MODEL_VERSION = "splink-fs-v1"

FEATURE_COLS = ["reg_norm", "type_family", "operator_icao", "flight_no_norm", "country_iso",
                "airport_icao", "lat", "lon", "damage", "fatalities", "event_date_local"]


@dataclass
class ProbConfig:
    date_window_days: int = 1
    accept_threshold: float = 0.95
    review_low: float = 0.30
    predict_threshold: float = 0.10
    min_t0_labels: int = 50
    em_record_sample_proportion: float = 1.0
    u_max_pairs: float = 2e6
    prior_match_rate_recall: float = 0.7
    save_model_path: str | None = "splink_model.json"
    extra_blocking_sql: list[str] = field(default_factory=list)


def _settings(cfg: ProbConfig):
    import splink.comparison_library as cl
    from splink import SettingsCreator
    from splink.blocking_rule_library import CustomRule

    w = cfg.date_window_days
    win = f"abs(date_diff('day', l.event_date_local, r.event_date_local)) <= {w}"
    blocking = [
        CustomRule(f"l.country_iso = r.country_iso AND {win}", sql_dialect="duckdb"),
        CustomRule(f"l.type_family = r.type_family AND {win}", sql_dialect="duckdb"),
        CustomRule(f"l.airport_icao = r.airport_icao AND abs(date_diff('day', l.event_date_local, r.event_date_local)) <= {w + 1}", sql_dialect="duckdb"),
        CustomRule("l.reg_norm = r.reg_norm AND abs(date_diff('day', l.event_date_local, r.event_date_local)) <= 7", sql_dialect="duckdb"),
    ] + [CustomRule(s, sql_dialect="duckdb") for s in cfg.extra_blocking_sql]

    comparisons = [
        cl.DamerauLevenshteinAtThresholds("reg_norm", [1, 2]),
        cl.AbsoluteDateDifferenceAtThresholds("event_date_local", input_is_string=False,
                                              metrics=["day", "day"], thresholds=[1, 3]),
        cl.ExactMatch("type_family").configure(term_frequency_adjustments=True),
        cl.ExactMatch("operator_icao").configure(term_frequency_adjustments=True),
        cl.ExactMatch("flight_no_norm"),
        cl.DistanceInKMAtThresholds("lat", "lon", [5, 25, 100]),
        cl.ExactMatch("airport_icao"),
        cl.ExactMatch("damage"),
        cl.ExactMatch("fatalities"),
    ]
    return SettingsCreator(
        link_type="link_only",
        unique_id_column_name="source_id",
        source_dataset_column_name="source",
        blocking_rules_to_generate_predictions=blocking,
        comparisons=comparisons,
        retain_intermediate_calculation_columns=True,
        retain_matching_columns=True,
    )


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["source", "source_id"] + FEATURE_COLS].copy()
    df["damage"] = df["damage"].where(df["damage"] != "UNK")
    df["event_date_local"] = pd.to_datetime(df["event_date_local"]).dt.date
    df["fatalities"] = df["fatalities"].astype("float64")
    return df


def run(wh: Warehouse, cfg: ProbConfig | None = None) -> pd.DataFrame:
    try:
        from splink import DuckDBAPI, Linker, block_on
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install splink>=4 to use the probabilistic layer") from e

    cfg = cfg or ProbConfig()
    recs = wh.records("event_date_local IS NOT NULL")
    sources = sorted(recs["source"].unique())
    if len(sources) < 2:
        log.warning("probabilistic layer needs ≥2 sources; have %s", sources)
        return pd.DataFrame()

    db_api = DuckDBAPI()  # separate in-memory DB; keeps Splink's temp tables out of the warehouse
    frames = [db_api.register(_prep(recs[recs["source"] == s]), dataset_display_name=s) for s in sources]
    linker = Linker(frames, _settings(cfg), log_level=30)

    # ---- training ---------------------------------------------------------
    linker.training.estimate_probability_two_random_records_match(
        [block_on("reg_norm", "event_date_local")], recall=cfg.prior_match_rate_recall)
    linker.training.estimate_u_using_random_sampling(max_pairs=cfg.u_max_pairs)

    t0 = wh.con.execute("""
        SELECT left_source AS source_dataset_l, left_id AS source_id_l,
               right_source AS source_dataset_r, right_id AS source_id_r
        FROM record_link WHERE method = 'T0' AND status IN ('auto_accept', 'accepted')
          AND left_source <> right_source
    """).df()
    trained = False
    if len(t0) >= cfg.min_t0_labels:
        try:
            labels = db_api.register(t0, dataset_display_name="t0_labels")
            linker.training.estimate_m_from_pairwise_labels(labels)
            trained = True
            log.info("m-probabilities estimated from %d T0 labels", len(t0))
        except Exception as e:  # noqa: BLE001
            log.warning("pairwise-label m estimation failed (%s); falling back to EM", e)
    if not trained:
        for rule in (block_on("reg_norm"),
                     block_on("type_family", "event_date_local")):
            try:
                linker.training.estimate_parameters_using_expectation_maximisation(
                    rule, record_sample_proportion=cfg.em_record_sample_proportion)
            except Exception as e:  # noqa: BLE001
                log.warning("EM session on %s failed: %s", rule, e)

    if cfg.save_model_path:
        linker.misc.save_model_to_json(cfg.save_model_path, overwrite=True)

    # ---- predict ----------------------------------------------------------
    pred = linker.inference.predict(threshold_match_probability=cfg.predict_threshold).as_pandas_dataframe()
    if pred.empty:
        return pred

    # orient pairs like record_link: (source||id) ascending
    l_key = pred["source_l"] + "\x1f" + pred["source_id_l"].astype(str)
    r_key = pred["source_r"] + "\x1f" + pred["source_id_r"].astype(str)
    swap = l_key > r_key
    out = pd.DataFrame({
        "left_source": pred["source_l"].where(~swap, pred["source_r"]),
        "left_id": pred["source_id_l"].where(~swap, pred["source_id_r"]).astype(str),
        "right_source": pred["source_r"].where(~swap, pred["source_l"]),
        "right_id": pred["source_id_r"].where(~swap, pred["source_id_l"]).astype(str),
        "score": pred["match_probability"].astype(float),
        "match_weight": pred["match_weight"].astype(float),
    })
    gamma_cols = [c for c in pred.columns if c.startswith("gamma_")]
    out["evidence"] = [
        json.dumps({"match_weight": round(mw, 3), **{c[6:]: int(g) for c, g in zip(gamma_cols, row)}})
        for mw, row in zip(out["match_weight"], pred[gamma_cols].itertuples(index=False))
    ]
    out = out[out["score"] >= cfg.review_low]
    out["status"] = ["auto_accept" if p >= cfg.accept_threshold else "needs_review" for p in out["score"]]

    c = wh.con
    c.execute("DELETE FROM record_link WHERE method = 'splink' AND decided_by = 'splink'")
    c.register("_pl", out)
    c.execute(f"""
        INSERT INTO record_link
        SELECT p.left_source, p.left_id, p.right_source, p.right_id, 'splink', p.score, p.evidence::JSON,
               p.status, '{MODEL_VERSION}', 'splink', now()
        FROM _pl p
        WHERE NOT EXISTS (SELECT 1 FROM record_link k
                          WHERE k.left_source = p.left_source AND k.left_id = p.left_id
                            AND k.right_source = p.right_source AND k.right_id = p.right_id)
    """)
    c.unregister("_pl")
    added = c.execute("SELECT status, count(*) n FROM record_link WHERE method='splink' GROUP BY 1").df()
    log.info("splink links added:\n%s", added.to_string(index=False))
    return out
