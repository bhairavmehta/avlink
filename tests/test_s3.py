"""S3 ingest against a moto-mocked bucket: listing, ETag caching, zip expansion, dispatch, failures."""
import io
import json
import zipfile

import boto3
import pytest
from moto import mock_aws

from avlink import s3 as avs3
from avlink.store import Warehouse

from .synthetic import make


@pytest.fixture
def bucket(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    with mock_aws():
        c = boto3.client("s3")
        c.create_bucket(Bucket="raw")
        syn = make(80, seed=9)
        c.put_object(Bucket="raw", Key="asn/part-0.csv", Body=syn.asn.iloc[:40].to_csv(index=False))
        c.put_object(Bucket="raw", Key="asn/part-1.jsonl",
                     Body="\n".join(json.dumps(r) for r in syn.asn.iloc[40:].fillna("").to_dict("records")))
        c.put_object(Bucket="raw", Key="asn/readme.txt", Body="ignore me")
        c.put_object(Bucket="raw", Key="asn/broken.json", Body="{not json")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("avh/2026.csv", syn.avherald.to_csv(index=False))
        c.put_object(Bucket="raw", Key="avherald/dump.zip", Body=buf.getvalue())  # zip unsupported for avherald
        c.put_object(Bucket="raw", Key="avherald/2026.csv", Body=syn.avherald.to_csv(index=False))
        yield c, syn


def test_ingest_asn_and_cache(bucket, tmp_path):
    c, syn = bucket
    wh = Warehouse(tmp_path / "w.duckdb")
    rep = avs3.ingest(wh, "asn", "s3://raw/asn/", cache_dir=tmp_path / "cache", s3=c)
    assert rep.objects == 4 and rep.files_loaded == 2
    assert rep.skipped == ["asn/readme.txt"] and len(rep.failed) == 1
    assert wh.stats().set_index("source").loc["asn", "n"] == len(syn.asn)
    # second run hits the ETag cache, upserts are idempotent
    rep2 = avs3.ingest(wh, "asn", "s3://raw/asn/", pattern="*.csv", cache_dir=tmp_path / "cache", s3=c)
    assert rep2.files_loaded == 1
    assert wh.stats().set_index("source").loc["asn", "n"] == len(syn.asn)


def test_ingest_avherald(bucket, tmp_path):
    c, syn = bucket
    wh = Warehouse(tmp_path / "w.duckdb")
    rep = avs3.ingest(wh, "avherald", "s3://raw/avherald/", cache_dir=tmp_path / "cache", s3=c)
    assert rep.records == len(syn.avherald) and rep.skipped == ["avherald/dump.zip"]
