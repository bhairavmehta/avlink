"""Aviation Safety Network (Flight Safety Foundation) loader.

Use a licensed export (FSF: "downloads or through an API" for members —
membership@flightsafety.org). Do not scrape aviation-safety.net / asn.flightsafety.org
for production use.

The export's column names aren't published, so `ASN_COLUMN_MAP` maps canonical
field → candidate column names; adjust it once you see the real file. Both the
curated Accident DB and the Wikibase can be loaded; pass `db="accident"` or
`db="wikibase"` so IDs don't collide (source_id = 'wb:271253' / 'db:20231105-0').
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import normalize as nz
from ..schema import conform

ASN_COLUMN_MAP: dict[str, list[str]] = {
    "id": ["id", "wikibase_id", "asn_id", "record_id", "wikiid"],
    "date": ["date", "Date", "accident_date"],
    "time": ["time", "Time"],
    "type": ["type", "Type", "aircraft_type", "acft_type"],
    "operator": ["operator", "Owner/operator", "owner_operator", "Operator"],
    "registration": ["registration", "Registration", "reg"],
    "msn": ["msn", "MSN", "c/n", "cn"],
    "fatalities": ["fatalities", "Fatalities", "fat"],
    "occupants": ["occupants", "Occupants"],
    "fat_occ": ["Fatalities/Occupants", "fatalities_occupants"],
    "damage": ["damage", "Aircraft damage", "aircraft_damage"],
    "category": ["category", "Category"],
    "location": ["location", "Location"],
    "country": ["country", "Country"],
    "phase": ["phase", "Phase"],
    "nature": ["nature", "Nature"],
    "dep": ["departure_airport", "Departure airport", "dep"],
    "dest": ["destination_airport", "Destination airport", "dest"],
    "agency": ["investigating_agency", "Investigating agency"],
    "report_no": ["report_number", "Report number", "report_no"],
    "narrative": ["narrative", "Narrative", "summary"],
    "sources": ["sources", "Sources", "source_urls"],
    "url": ["url", "URL", "link"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "lng", "longitude"],
}

_AIRPORT_CODES = re.compile(r"\(([A-Z0-9]{3})?\s*/?\s*([A-Z]{4})\)")


def read_export(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".json", ".jsonl", ".ndjson"):
        with open(path, encoding="utf-8") as fh:
            if path.suffix.lower() == ".json":
                data = json.load(fh)
                data = data if isinstance(data, list) else data.get("records") or data.get("data") or []
            else:
                data = [json.loads(line) for line in fh if line.strip()]
        return pd.DataFrame(data)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def _pick(row: pd.Series, key: str):
    for c in ASN_COLUMN_MAP.get(key, [key]):
        if c in row.index and not nz.is_null(row[c]):
            return row[c]
    return None


def _airport_icao(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = _AIRPORT_CODES.search(str(text))
    return m.group(2) if m else None


def _fat_occ(row: pd.Series) -> tuple[Optional[int], Optional[int]]:
    f, o = _pick(row, "fatalities"), _pick(row, "occupants")
    combo = _pick(row, "fat_occ")
    if combo and (f is None or o is None):
        m = re.search(r"(\d+)\s*/\s*(?:Occupants:\s*)?(\d+)", str(combo))
        if m:
            f, o = f or m.group(1), o or m.group(2)
    if f is not None:
        m = re.search(r"Fatalities:\s*(\d+)\s*/\s*Occupants:\s*(\d+)", str(f))
        if m:
            f, o = m.group(1), m.group(2)
    to_int = lambda v: int(re.sub(r"\D", "", str(v))) if v is not None and re.search(r"\d", str(v)) else None
    return to_int(f), to_int(o)


def to_canonical(raw: pd.DataFrame, db: str = "wikibase") -> pd.DataFrame:
    prefix = {"wikibase": "wb", "accident": "db"}.get(db, db)
    rows = []
    for _, r in raw.iterrows():
        rid = nz.clean_str(_pick(r, "id"))
        url = nz.clean_str(_pick(r, "url"))
        if not rid and url:
            m = re.search(r"/(?:wikibase|record\.php\?id=)/?([\w\-]+)", url)
            rid = m.group(1) if m else url
        if not rid:
            continue
        location = nz.clean_str(_pick(r, "location"))
        country = nz.country_iso(_pick(r, "country")) or nz.country_iso(location)
        d = nz.parse_loose_date(_pick(r, "date"))
        acft_type = nz.clean_str(_pick(r, "type"))
        operator = nz.clean_str(_pick(r, "operator"))
        dep, dest = nz.clean_str(_pick(r, "dep")), nz.clean_str(_pick(r, "dest"))
        narrative = nz.clean_str(_pick(r, "narrative"))
        sources = _pick(r, "sources")
        sources_txt = " ".join(sources) if isinstance(sources, list) else (nz.clean_str(sources) or "")
        report_no = nz.clean_str(_pick(r, "report_no"))
        fat, occ = _fat_occ(r)
        rows.append(dict(
            source="asn",
            source_id=f"{prefix}:{rid}",
            event_date_local=d,
            event_ts_utc=None,  # ASN times are local and ASN carries no tz; enrich from airport tz if needed
            tz=None,
            reg_raw=nz.clean_str(_pick(r, "registration")),
            reg_norm=nz.norm_reg(_pick(r, "registration"), country),
            msn=nz.norm_msn(_pick(r, "msn")),
            make_model_raw=acft_type,
            icao_type=nz.icao_designator(acft_type),
            type_family=nz.type_family(acft_type),
            operator_raw=operator,
            operator_icao=nz.operator_icao(operator),
            flight_no_norm=nz.norm_flight_no(_flight_from_text(narrative)),
            country_iso=country,
            location_raw=location,
            airport_icao=_airport_icao(location) or _airport_icao(dep) or _airport_icao(dest),
            dep_icao=_airport_icao(dep),
            dest_icao=_airport_icao(dest),
            lat=nz.parse_coord(_pick(r, "lat"), True),
            lon=nz.parse_coord(_pick(r, "lon"), False),
            fatalities=fat, occupants=occ,
            damage=nz.norm_damage(_pick(r, "damage")),
            phase=(nz.clean_str(_pick(r, "phase")) or "").lower() or None,
            category=nz.norm_category(_pick(r, "category")),
            narrative=narrative,
            ntsb_no_refs=nz.extract_ntsb_numbers(report_no, narrative, sources_txt),
            docket_refs=nz.extract_docket_ids(sources_txt, narrative),
            url=url,
        ))
    return conform(pd.DataFrame(rows))


_FLIGHT_IN_TEXT = re.compile(r"\bflight\s+(?:number\s+)?([A-Z0-9]{2,3}[\s\-]?\d{1,4})\b", re.I)


def _flight_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = _FLIGHT_IN_TEXT.search(text)
    return m.group(1) if m else None
