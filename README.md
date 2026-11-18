# avlink — link NTSB, ASN and Aviation Herald records

Reads your already-downloaded files, normalizes them into one canonical schema in DuckDB,
links records describing the same occurrence, clusters them into unified events, and
exports one feature row per event.

```
pip install -e ".[splink,s3]"          # + ".[llm]" for Claude adjudication; apt-get install mdbtools for avall.mdb
```

## Run
```
# From S3 (boto3 credential chain; incremental via ETag cache in .avlink_cache/)
avlink --db av.duckdb s3 ntsb     s3://bucket/raw/ntsb/       # .mdb, .zip(avall), CAROL .json/.csv
avlink --db av.duckdb s3 asn      s3://bucket/raw/asn/        # .csv/.json/.jsonl/.parquet
avlink --db av.duckdb s3 avherald s3://bucket/raw/avherald/   # .csv/.jsonl
# ...or from local files:
avlink --db av.duckdb ntsb-mdb   avall.mdb up15NOV.mdb     # or: ntsb-csv carol_export.json
avlink --db av.duckdb faa        ReleasableAircraft/        # optional: MSN/type enrichment for N-numbers
avlink --db av.duckdb airports   airports.csv               # optional: OurAirports, FAA LID → ICAO, lat/lon
avlink --db av.duckdb asn        asn_export.csv --asn-db wikibase
avlink --db av.duckdb avherald   avherald.csv               # url, headline, [first_paragraph]
avlink --db av.duckdb link                                  # enrich → T0–T3 rules → Splink
avlink --db av.duckdb adjudicate --llm anthropic            # grey zone only; --llm heuristic = offline dry run
avlink --db av.duckdb cluster
avlink --db av.duckdb sample todo.csv   → label is_match 1/0 →   avlink labels done.csv
avlink --db av.duckdb evaluate
avlink --db av.duckdb features events.parquet
avlink --db av.duckdb wide avh_asn_ntsb.parquet         # AvHerald ⟗ ASN ⟗ NTSB, one row per match
```

## Linkage tiers
| Tier | Rule | Default status |
|---|---|---|
| T0 | shared NTSB number or docket ProjectID/mkey (regex-mined from ASN report no./sources/narrative) | auto_accept (review if Δdate > 3d) |
| T1 | same normalized registration, \|Δdate\| ≤ 1 | auto_accept |
| T2 | registration Damerau-Levenshtein = 1, \|Δdate\| ≤ 1, same type family | auto_accept (review if Δdate > 0) |
| T3 | same flight number, \|Δdate\| ≤ 1, no type conflict | auto_accept |
| SAME_EVENT | NTSB aircraft rows sharing ev_id (mid-airs) | auto_accept |
| splink | Fellegi-Sunter on residuals; m from T0 labels when ≥50, else EM | ≥0.95 accept, 0.30–0.95 review |
| llm | select-among-candidates, 2 runs, unanimous only | accepted / rejected / review |

Conflict guards (type family, MSN, >300 km, operator for T3) demote any rule hit to `needs_review`.
Human labels (`avlink labels`) are final and never overwritten by re-runs.

## Adapting to your files
* **ASN**: edit `ASN_COLUMN_MAP` in `avlink/sources/asn.py` to your export's headers.
* **NTSB**: avall.mdb column names are handled natively; CAROL/API aliases are in `_API_ALIASES`.
* **AvHerald**: `avlink/sources/avherald.py` parses headline + first paragraph. Article text is
  dropped unless `--keep-text`; only use that if your licence covers storing/AI use.
* Reference tables: set `AVLINK_TYPES_CSV`, `AVLINK_AIRLINES_CSV`, `AVLINK_COUNTRIES_CSV` to extend
  the seeded ICAO type / airline / country maps.

## Tests
`pytest -q` — unit tests plus an end-to-end run on synthetic data with injected noise
(registration typos, ±1 day shifts, missing N-prefix, mid-airs, conflicting re-used marks, flight-number distractors).
