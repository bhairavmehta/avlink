"""S3 ingest: list objects under a prefix, cache them locally, dispatch to source loaders.

    avlink --db av.duckdb s3 ntsb     s3://my-bucket/raw/ntsb/
    avlink --db av.duckdb s3 asn      s3://my-bucket/raw/asn/      --asn-db wikibase
    avlink --db av.duckdb s3 avherald s3://my-bucket/raw/avherald/

Credentials come from the standard boto3 chain (env vars, ~/.aws, SSO, instance role).
Set AWS_PROFILE / AWS_REGION as usual; AVLINK_S3_ENDPOINT for MinIO / S3-compatible stores.

Caching: objects are downloaded to `cache_dir/<bucket>/<key>` and re-downloaded only
when the ETag changes (tracked in a sidecar .etag file), so re-runs are incremental.
.zip objects (e.g. NTSB avall.zip) are extracted next to the cached file.

Dispatch by file type per source:
  ntsb      .mdb (needs mdbtools) · .json/.jsonl (CAROL/API) · .csv (CAROL) · .zip containing .mdb
  asn       .csv · .json · .jsonl · .parquet
  avherald  .csv · .jsonl   (url, headline, [first_paragraph])
Per-object failures are logged and skipped; a summary is returned.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlparse

import pandas as pd

from .sources import asn, avherald, ntsb
from .store import Warehouse

log = logging.getLogger(__name__)

SUPPORTED = {
    "ntsb": (".mdb", ".json", ".jsonl", ".ndjson", ".csv", ".zip"),
    "asn": (".csv", ".json", ".jsonl", ".ndjson", ".parquet"),
    "avherald": (".csv", ".jsonl", ".ndjson"),
}


def parse_uri(uri: str) -> tuple[str, str]:
    u = urlparse(uri)
    if u.scheme not in ("s3", "s3a"):
        raise ValueError(f"not an s3 URI: {uri}")
    return u.netloc, u.path.lstrip("/")


def client():
    import boto3
    return boto3.client("s3", endpoint_url=os.getenv("AVLINK_S3_ENDPOINT") or None)


def list_objects(uri: str, pattern: str = "*", s3=None) -> Iterator[dict]:
    s3 = s3 or client()
    bucket, prefix = parse_uri(uri)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/") or not fnmatch.fnmatch(Path(key).name, pattern):
                continue
            yield {"bucket": bucket, **obj}


def fetch(obj: dict, cache_dir: Path, s3=None) -> Path:
    s3 = s3 or client()
    dest = cache_dir / obj["bucket"] / obj["Key"]
    etag_file = dest.with_name(dest.name + ".etag")
    etag = obj.get("ETag", "").strip('"')
    if dest.exists() and etag_file.exists() and etag_file.read_text() == etag:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    s3.download_file(obj["bucket"], obj["Key"], str(tmp))
    tmp.replace(dest)
    etag_file.write_text(etag)
    return dest


def _expand_zip(path: Path) -> list[Path]:
    out_dir = path.with_suffix("")
    out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for m in z.infolist():
            target = (out_dir / m.filename).resolve()
            if not str(target).startswith(str(out_dir.resolve())) or m.is_dir():
                continue  # zip-slip guard
            z.extract(m, out_dir)
    return sorted(p for p in out_dir.rglob("*") if p.is_file())


def _read_json_records(path: Path) -> list[dict]:
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("value") or data.get("Results") or data.get("results") or data.get("data") or [data]
    return data


def to_canonical(source: str, path: Path, asn_db: str = "wikibase", keep_text: bool = False) -> pd.DataFrame:
    sfx = path.suffix.lower()
    if source == "ntsb":
        if sfx == ".mdb":
            return ntsb.to_canonical(*ntsb.export_mdb_tables(str(path)))
        if sfx in (".json", ".jsonl", ".ndjson"):
            return ntsb.to_canonical(*ntsb.api_cases_to_frames(_read_json_records(path)))
        if sfx == ".csv":
            raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
            return ntsb.to_canonical(*ntsb.api_cases_to_frames(raw.to_dict("records")))
    elif source == "asn":
        return asn.to_canonical(asn.read_export(path), db=asn_db)
    elif source == "avherald":
        return avherald.to_canonical(avherald.read_extracts(path), keep_text=keep_text)
    raise ValueError(f"unsupported file for {source}: {path.name}")


@dataclass
class IngestReport:
    objects: int = 0
    files_loaded: int = 0
    records: int = 0
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def ingest(wh: Warehouse, source: str, uri: str, pattern: str = "*",
           cache_dir: str | Path = ".avlink_cache", asn_db: str = "wikibase",
           keep_text: bool = False, s3=None, batch_records: int = 50_000) -> IngestReport:
    """Load every supported object under `uri` into the warehouse.

    Objects are processed in key order, so name NTSB change files so they sort after
    avall (NTSB's upDDMON names don't sort by date — prefix them, e.g. 2026-11-15_up15NOV.zip).
    """
    if source not in SUPPORTED:
        raise ValueError(f"source must be one of {list(SUPPORTED)}")
    s3 = s3 or client()
    cache = Path(cache_dir)
    rep = IngestReport()
    buf: list[pd.DataFrame] = []
    buffered = 0

    def flush():
        nonlocal buf, buffered
        if buf:
            rep.records += wh.upsert_records(pd.concat(buf, ignore_index=True))
            buf, buffered = [], 0

    for obj in sorted(list_objects(uri, pattern, s3), key=lambda o: o["Key"]):
        rep.objects += 1
        key = obj["Key"]
        if not key.lower().endswith(SUPPORTED[source]):
            rep.skipped.append(key)
            continue
        try:
            local = fetch(obj, cache, s3)
            paths = _expand_zip(local) if local.suffix.lower() == ".zip" else [local]
            for p in paths:
                if not p.name.lower().endswith(SUPPORTED[source]) or p.suffix.lower() == ".zip":
                    continue
                df = to_canonical(source, p, asn_db, keep_text)
                rep.files_loaded += 1
                if p.suffix.lower() == ".mdb":
                    flush()                      # keep avall → change-file ordering exact
                    rep.records += wh.upsert_records(df)
                else:
                    buf.append(df)
                    buffered += len(df)
                    if buffered >= batch_records:
                        flush()
        except Exception as e:  # noqa: BLE001
            log.exception("failed on s3://%s/%s", obj["bucket"], key)
            rep.failed.append((key, f"{type(e).__name__}: {e}"))
    flush()
    log.info("s3 ingest %s: %d objects, %d files, %d records, %d skipped, %d failed",
             source, rep.objects, rep.files_loaded, rep.records, len(rep.skipped), len(rep.failed))
    return rep
