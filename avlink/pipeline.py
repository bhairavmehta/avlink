"""Command-line entry point:  python -m avlink.pipeline <command> [...]

Typical first run
-----------------
  avlink init      --db avlink.duckdb
  avlink s3 ntsb   --db avlink.duckdb  s3://bucket/raw/ntsb/      # or ntsb-mdb / ntsb-csv for local files
  avlink ntsb-mdb  --db avlink.duckdb  data/avall.mdb  [data/up15NOV.mdb ...]
  avlink faa       --db avlink.duckdb  data/ReleasableAircraft/
  avlink airports  --db avlink.duckdb  data/airports.csv
  avlink asn       --db avlink.duckdb  data/asn_wikibase_export.csv --asn-db wikibase
  avlink avherald  --db avlink.duckdb  data/avherald_permitted.csv
  avlink link      --db avlink.duckdb               # enrich + T0–T3 + Splink
  avlink adjudicate --db avlink.duckdb --llm anthropic   # or --llm heuristic for a dry run
  avlink cluster   --db avlink.duckdb
  avlink sample    --db avlink.duckdb  labels_todo.csv
  avlink labels    --db avlink.duckdb  labels_done.csv
  avlink evaluate  --db avlink.duckdb
  avlink features  --db avlink.duckdb  events_features.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date

import pandas as pd

from . import adjudicate, cluster, deterministic, evaluate, features, probabilistic, reference, s3
from .sources import asn, avherald, faa_registry, ntsb
from .store import Warehouse

log = logging.getLogger("avlink")


def _load_reference_overrides():
    for env, fn in (("AVLINK_TYPES_CSV", reference.load_type_designators),
                    ("AVLINK_AIRLINES_CSV", reference.load_airlines),
                    ("AVLINK_COUNTRIES_CSV", reference.load_countries)):
        if os.getenv(env):
            n = fn(os.environ[env])
            log.info("loaded %d rows from %s", n, os.environ[env])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="avlink", description="Link NTSB, ASN and AvHerald records")
    ap.add_argument("--db", default=os.getenv("AVLINK_DB", "avlink.duckdb"))
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    p = sub.add_parser("ntsb-mdb"); p.add_argument("mdb", nargs="+")
    p = sub.add_parser("ntsb-api"); p.add_argument("--start", required=True); p.add_argument("--end", default=str(date.today()))
    p = sub.add_parser("ntsb-csv", help="CAROL CSV/JSON export"); p.add_argument("path")
    p = sub.add_parser("faa"); p.add_argument("folder")
    p = sub.add_parser("airports"); p.add_argument("csv")
    p = sub.add_parser("asn"); p.add_argument("path"); p.add_argument("--asn-db", default="wikibase", choices=["wikibase", "accident"])
    p = sub.add_parser("avherald"); p.add_argument("path"); p.add_argument("--keep-text", action="store_true",
                                                                         help="ONLY if your AvHerald licence covers text/AI use")
    p = sub.add_parser("link"); p.add_argument("--no-splink", action="store_true"); p.add_argument("--window", type=int, default=1)
    p = sub.add_parser("adjudicate"); p.add_argument("--llm", choices=["anthropic", "heuristic"], default="heuristic")
    p.add_argument("--model", default="claude-sonnet-5-5"); p.add_argument("--limit", type=int)
    p.add_argument("--allow-avherald-text", action="store_true")
    sub.add_parser("cluster")
    p = sub.add_parser("sample"); p.add_argument("out_csv"); p.add_argument("--per-stratum", type=int, default=40)
    p = sub.add_parser("labels"); p.add_argument("csv"); p.add_argument("--labeller", default="analyst")
    sub.add_parser("evaluate")
    p = sub.add_parser("review"); p.add_argument("out_csv")
    p = sub.add_parser("features"); p.add_argument("out")
    sub.add_parser("stats")
    p = sub.add_parser("s3", help="ingest every supported object under an s3:// prefix")
    p.add_argument("source", choices=["ntsb", "asn", "avherald"]); p.add_argument("uri")
    p.add_argument("--pattern", default="*", help="filename glob, e.g. '*.jsonl'")
    p.add_argument("--cache-dir", default=os.getenv("AVLINK_CACHE", ".avlink_cache"))
    p.add_argument("--asn-db", default="wikibase", choices=["wikibase", "accident"])
    p.add_argument("--keep-text", action="store_true")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _load_reference_overrides()
    pd.set_option("display.width", 200)
    wh = Warehouse(a.db)
    try:
        if a.cmd == "init":
            print(f"initialised {a.db}")
        elif a.cmd == "ntsb-mdb":
            for path in a.mdb:  # avall first, then change files in date order
                wh.upsert_records(ntsb.to_canonical(*ntsb.export_mdb_tables(path)))
        elif a.cmd == "ntsb-api":
            cases = ntsb.NtsbApiClient().cases_by_date_range(pd.Timestamp(a.start).date(), pd.Timestamp(a.end).date())
            wh.upsert_records(ntsb.to_canonical(*ntsb.api_cases_to_frames(cases)))
        elif a.cmd == "ntsb-csv":
            raw = pd.read_json(a.path) if a.path.endswith(".json") else pd.read_csv(a.path, dtype=str)
            wh.upsert_records(ntsb.to_canonical(*ntsb.api_cases_to_frames(raw.to_dict("records"))))
        elif a.cmd == "faa":
            wh.load_faa_registry(faa_registry.load_registry(a.folder))
        elif a.cmd == "airports":
            wh.load_airports(a.csv)
        elif a.cmd == "asn":
            wh.upsert_records(asn.to_canonical(asn.read_export(a.path), db=a.asn_db))
        elif a.cmd == "avherald":
            wh.upsert_records(avherald.to_canonical(avherald.read_extracts(a.path), keep_text=a.keep_text))
        elif a.cmd == "link":
            wh.enrich()
            print(deterministic.run(wh, deterministic.DetConfig(date_window_days=a.window)).to_string(index=False))
            if not a.no_splink:
                probabilistic.run(wh, probabilistic.ProbConfig(date_window_days=a.window))
        elif a.cmd == "adjudicate":
            fn = adjudicate.anthropic_llm_fn(a.model) if a.llm == "anthropic" else adjudicate.heuristic_llm_fn
            cfg = adjudicate.AdjConfig(limit_anchors=a.limit, allow_avherald_text=a.allow_avherald_text)
            adjudicate.run(wh, fn, cfg)
        elif a.cmd == "cluster":
            cluster.run(wh)
        elif a.cmd == "sample":
            evaluate.sample_for_labelling(wh, a.out_csv, a.per_stratum)
        elif a.cmd == "labels":
            print(f"imported {evaluate.import_labels(wh, a.csv, a.labeller)} labels")
        elif a.cmd == "evaluate":
            print(evaluate.metrics(wh).to_string(index=False))
            print(evaluate.blocking_recall_vs_t0(wh).to_string(index=False))
        elif a.cmd == "review":
            print(f"{len(evaluate.review_queue(wh, a.out_csv))} pairs need review → {a.out_csv}")
        elif a.cmd == "features":
            print(f"{len(features.export(wh, a.out))} events → {a.out}")
        elif a.cmd == "s3":
            rep = s3.ingest(wh, a.source, a.uri, a.pattern, a.cache_dir, a.asn_db, a.keep_text)
            print(f"{rep.objects} objects, {rep.files_loaded} files, {rep.records} records, "
                  f"{len(rep.skipped)} skipped, {len(rep.failed)} failed")
            for k, err in rep.failed[:20]:
                print(f"  FAILED {k}: {err}")
        elif a.cmd == "stats":
            print(wh.stats().to_string(index=False))
            print(wh.con.execute("SELECT method, status, count(*) n FROM record_link GROUP BY ALL ORDER BY 1,2").df().to_string(index=False))
    finally:
        wh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
