"""End-to-end: synthetic sources → warehouse → links → clusters, scored against ground truth."""
import itertools
import json

import pandas as pd
import pytest

from avlink import wide, adjudicate, cluster, deterministic, evaluate, features, probabilistic
from avlink.sources import asn, avherald, ntsb
from avlink.store import Warehouse

from .synthetic import make


def pairwise_scores(wh: Warehouse, truth: dict) -> dict:
    """Precision/recall of the final clustering over all cross-record pairs."""
    mem = wh.con.execute("SELECT * FROM event_member").df()
    pred_pairs, true_pairs = set(), set()
    for _, g in mem.groupby("unified_event_id"):
        keys = sorted(zip(g.source, g.source_id))
        pred_pairs.update(itertools.combinations(keys, 2))
    by_true: dict[str, list] = {}
    for k, gid in truth.items():
        by_true.setdefault(gid, []).append(k)
    for keys in by_true.values():
        true_pairs.update(itertools.combinations(sorted(keys), 2))
    tp = len(pred_pairs & true_pairs)
    return dict(tp=tp, fp=len(pred_pairs - true_pairs), fn=len(true_pairs - pred_pairs),
                precision=tp / max(1, len(pred_pairs)), recall=tp / max(1, len(true_pairs)))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    syn = make(n_events=600, seed=3)
    db = tmp_path_factory.mktemp("db") / "t.duckdb"
    wh = Warehouse(db)
    wh.upsert_records(ntsb.to_canonical(syn.events, syn.aircraft, syn.narratives))
    wh.upsert_records(asn.to_canonical(syn.asn, db="wikibase"))
    wh.upsert_records(avherald.to_canonical(syn.avherald))
    yield wh, syn
    wh.close()


def test_load_counts(built):
    wh, syn = built
    st = wh.stats().set_index("source")
    assert st.loc["ntsb", "n"] == len(syn.aircraft)
    assert st.loc["asn", "n"] == len(syn.asn)
    assert st.loc["avherald", "n"] == len(syn.avherald)
    assert st.loc["ntsb", "pct_reg"] == 1.0  # leading-N restored


def test_deterministic_and_conflicts(built):
    wh, syn = built
    summary = deterministic.run(wh)
    assert {"T0", "T1"} <= set(summary["method"])
    # conflict guard: same reg, different type family must never be auto-accepted
    bad = wh.con.execute("""
        SELECT count(*) FROM record_link
        WHERE status='auto_accept' AND method IN ('T1','T2','T3')
          AND json_array_length(evidence->'conflicts') > 0""").fetchone()[0]
    assert bad == 0


def test_full_pipeline_quality(built):
    wh, syn = built
    probabilistic.run(wh, probabilistic.ProbConfig(save_model_path=None))
    adjudicate.run(wh, adjudicate.heuristic_llm_fn)
    cluster.run(wh)
    s = pairwise_scores(wh, syn.truth)
    print("pairwise:", s)
    assert s["precision"] >= 0.97, s
    assert s["recall"] >= 0.90, s
    # mid-air: both aircraft share one unified event
    ev = wh.con.execute("SELECT * FROM event WHERE len(ntsb_ev_ids) = 1 AND n_records >= 3").df()
    assert len(ev) > 0


def test_eval_and_features(built, tmp_path):
    wh, syn = built
    sheet = evaluate.sample_for_labelling(wh, tmp_path / "todo.csv", per_stratum=10)
    # label with ground truth, as an analyst would
    sheet["is_match"] = [
        int(syn.truth.get((a, b)) == syn.truth.get((c, d)) and syn.truth.get((a, b)) is not None)
        for a, b, c, d in zip(sheet.left_source, sheet.left_id, sheet.right_source, sheet.right_id)]
    sheet.to_csv(tmp_path / "done.csv", index=False)
    assert evaluate.import_labels(wh, tmp_path / "done.csv") == len(sheet)
    m = evaluate.metrics(wh)
    print(m.to_string(index=False))
    assert m.set_index(["pair", "method"]).loc[("ALL", "ALL"), "precision"] >= 0.9
    print(evaluate.blocking_recall_vs_t0(wh).to_string(index=False))
    cluster.run(wh)
    df = features.export(wh, tmp_path / "f.parquet")
    assert df["unified_event_id"].is_unique
    assert {"in_ntsb", "in_asn", "in_avherald", "min_link_score"} <= set(df.columns)


def test_wide_outer_join(built):
    wh, syn = built
    df = wide.build(wh)
    # every record of every source appears at least once (outer join loses nothing)
    for src, col in (("avherald", "avh_source_id"), ("asn", "asn_source_id"), ("ntsb", "ntsb_source_id")):
        ids = set(wh.records(f"source = '{src}'")["source_id"])
        assert ids <= set(df[col].dropna()), src
    # co-occurring ids in a row must be truly the same event
    bad = 0
    for _, r in df.iterrows():
        keys = [(s, r[c]) for s, c in (("avherald", "avh_source_id"), ("asn", "asn_source_id"), ("ntsb", "ntsb_source_id"))
                if isinstance(r[c], str)]
        if len({syn.truth.get(k) for k in keys}) > 1:
            bad += 1
    assert bad / len(df) < 0.01
    assert {"AVH+ASN+NTSB", "ASN+NTSB", "AVH+ASN", "AVH"} <= set(df["match_pattern"])
