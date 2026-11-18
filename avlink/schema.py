"""Canonical record schema and warehouse DDL.

Every source loader emits a pandas DataFrame with exactly CANONICAL_COLUMNS.
Linkage, clustering and evaluation only ever see this shape.
"""
from __future__ import annotations

import pandas as pd

# (column, pandas dtype, description)
CANONICAL_SPEC: list[tuple[str, str, str]] = [
    ("source", "string", "ntsb | asn | avherald"),
    ("source_id", "string", "native ID: ntsb_no (or ev_id|acft_key), ASN wikibase/db id, AvHerald article id"),
    ("event_group_id", "string", "source-native event grouping (NTSB ev_id); null elsewhere"),
    ("snapshot_ts", "datetime64[ns]", "ingest snapshot time (UTC)"),
    ("event_date_local", "datetime64[ns]", "occurrence date, local to the event"),
    ("event_ts_utc", "datetime64[ns]", "occurrence timestamp in UTC when time+tz known"),
    ("tz", "string", "raw time-zone code"),
    ("reg_raw", "string", "registration as published"),
    ("reg_norm", "string", "normalized registration (upper, no separators, US marks with N)"),
    ("msn", "string", "manufacturer serial number, normalized"),
    ("make_model_raw", "string", "make/model text as published"),
    ("icao_type", "string", "ICAO Doc 8643 designator when known"),
    ("type_family", "string", "coarse type family used for blocking/gating"),
    ("operator_raw", "string", "operator text as published"),
    ("operator_icao", "string", "ICAO 3-letter airline code when resolvable"),
    ("flight_no_norm", "string", "carrier code + flight number, e.g. UA1854"),
    ("country_iso", "string", "ISO 3166-1 alpha-2"),
    ("location_raw", "string", "location text as published"),
    ("airport_icao", "string", "nearest/related airport ICAO code"),
    ("dep_icao", "string", "departure airport ICAO"),
    ("dest_icao", "string", "destination airport ICAO"),
    ("lat", "float64", "decimal latitude"),
    ("lon", "float64", "decimal longitude"),
    ("fatalities", "Int64", "total fatalities"),
    ("occupants", "Int64", "persons on board"),
    ("damage", "string", "DEST | SUBS | MINR | NONE | UNK"),
    ("phase", "string", "phase of flight (free text, lowercased)"),
    ("category", "string", "accident | incident | crash | news | report | other"),
    ("narrative", "string", "narrative text (respect source licence before storing!)"),
    ("ntsb_no_refs", "object", "list[str] NTSB numbers this record IS or REFERENCES"),
    ("docket_refs", "object", "list[str] NTSB docket ProjectIDs / mkeys this record IS or REFERENCES"),
    ("url", "string", "canonical URL of the record"),
]
CANONICAL_COLUMNS = [c for c, _, _ in CANONICAL_SPEC]

DAMAGE_LEVELS = ("DEST", "SUBS", "MINR", "NONE", "UNK")


def empty_canonical() -> pd.DataFrame:
    return conform(pd.DataFrame(columns=CANONICAL_COLUMNS))


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing columns, drop extras, coerce dtypes, order columns."""
    df = df.copy()
    for col, dtype, _ in CANONICAL_SPEC:
        if col not in df.columns:
            df[col] = None
    df = df[CANONICAL_COLUMNS]
    for col, dtype, _ in CANONICAL_SPEC:
        if dtype == "object":
            df[col] = df[col].apply(lambda v: list(v) if isinstance(v, (list, tuple, set)) else [])
        elif dtype.startswith("datetime"):
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
            if getattr(df[col].dt, "tz", None) is not None:
                df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
            df[col] = df[col].astype("datetime64[ns]")
        elif dtype == "Int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif dtype == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        else:
            df[col] = df[col].astype("string")
            df[col] = df[col].where(df[col].str.strip().ne(""), pd.NA)
    if df["snapshot_ts"].isna().any():
        df["snapshot_ts"] = df["snapshot_ts"].fillna(pd.Timestamp.now('UTC').tz_localize(None).floor("s"))
    return df


DDL = """
CREATE TABLE IF NOT EXISTS source_record (
    source VARCHAR, source_id VARCHAR, event_group_id VARCHAR, snapshot_ts TIMESTAMP,
    event_date_local DATE, event_ts_utc TIMESTAMP, tz VARCHAR,
    reg_raw VARCHAR, reg_norm VARCHAR, msn VARCHAR, make_model_raw VARCHAR,
    icao_type VARCHAR, type_family VARCHAR, operator_raw VARCHAR, operator_icao VARCHAR,
    flight_no_norm VARCHAR, country_iso VARCHAR, location_raw VARCHAR,
    airport_icao VARCHAR, dep_icao VARCHAR, dest_icao VARCHAR, lat DOUBLE, lon DOUBLE,
    fatalities BIGINT, occupants BIGINT, damage VARCHAR, phase VARCHAR, category VARCHAR,
    narrative VARCHAR, ntsb_no_refs VARCHAR[], docket_refs VARCHAR[], url VARCHAR,
    PRIMARY KEY (source, source_id)
);

-- Full history of every snapshot (bronze-ish audit trail; records get edited upstream).
CREATE TABLE IF NOT EXISTS source_record_history AS SELECT * FROM source_record WHERE false;

CREATE TABLE IF NOT EXISTS faa_registry (
    reg_norm VARCHAR, serial_no VARCHAR, mfr VARCHAR, model VARCHAR, type_family VARCHAR,
    valid_from DATE, valid_to DATE, is_current BOOLEAN
);

CREATE TABLE IF NOT EXISTS airport (
    icao VARCHAR, iata VARCHAR, local_code VARCHAR, name VARCHAR, municipality VARCHAR,
    iso_country VARCHAR, lat DOUBLE, lon DOUBLE
);

CREATE TABLE IF NOT EXISTS record_link (
    left_source VARCHAR, left_id VARCHAR, right_source VARCHAR, right_id VARCHAR,
    method VARCHAR,          -- T0 | T1 | T2 | T3 | SAME_EVENT | splink | llm | manual
    score DOUBLE,            -- rule confidence or match probability
    evidence JSON,           -- field agreements, edit distance, delta days, LLM rationale
    status VARCHAR,          -- auto_accept | needs_review | accepted | rejected
    model_version VARCHAR, decided_by VARCHAR, decided_at TIMESTAMP,
    PRIMARY KEY (left_source, left_id, right_source, right_id)
);

CREATE TABLE IF NOT EXISTS event (
    unified_event_id VARCHAR PRIMARY KEY, canonical_date DATE, canonical_reg VARCHAR,
    canonical_type_family VARCHAR, canonical_operator VARCHAR, ntsb_ev_ids VARCHAR[],
    n_sources INTEGER, n_records INTEGER, max_fatalities BIGINT, worst_damage VARCHAR,
    min_link_score DOUBLE, link_methods VARCHAR[], flags VARCHAR[]
);

CREATE TABLE IF NOT EXISTS event_member (
    unified_event_id VARCHAR, source VARCHAR, source_id VARCHAR
);

CREATE TABLE IF NOT EXISTS link_label (
    left_source VARCHAR, left_id VARCHAR, right_source VARCHAR, right_id VARCHAR,
    is_match BOOLEAN, labeller VARCHAR, labelled_at TIMESTAMP, note VARCHAR
);
"""
