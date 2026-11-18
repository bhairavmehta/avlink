"""The Aviation Herald — parser for PERMITTED extracts only.

!!! Licensing !!!
avherald.com's footer states "reproduction, redistribution and AI learning/use
prohibited" and its robots rules block crawlers. This module therefore contains
NO fetching/scraping code. It parses records you are entitled to hold, e.g.:

  * a data/AI-use licence granted in writing by the publisher, or
  * your own manually curated list of article URLs + headlines.

By default only extracted *facts* (registration, date, flight, type, operator,
POB) are kept and `narrative` is dropped. Set `keep_text=True` only if your
licence explicitly covers storing/processing article text.

Input: CSV/JSONL with columns  url | article_id, headline, [first_paragraph], [created]
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import normalize as nz
from .. import reference as refs
from ..schema import conform

HEADLINE_RE = re.compile(
    r"^(?P<cat>Crash|Accident|Incident|News|Report):\s+"
    r"(?P<op>.+?)\s+(?P<type>[A-Z][A-Z0-9]{1,3})\s+"
    r"(?P<prep>at|near|over|enroute|en route|on approach to|on departure from)\s+"
    r"(?P<place>.+?)\s+on\s+(?P<date>[A-Z][a-z]{2}\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4})"
    r"(?:,\s*(?P<cause>.*))?$"
)
REG_RE = re.compile(r"\bregistration\s+([A-Z0-9][A-Z0-9\-]{1,9})\b", re.I)
FLIGHT_RE = re.compile(r"\bflight\s+([A-Z0-9]{2,3}-?\d{1,4}[A-Z]?)\b", re.I)
ROUTE_RE = re.compile(r"\bfrom\s+(?P<dep>.+?)\s+to\s+(?P<dest>.+?)\s+with\b", re.I)
POB_RE = re.compile(r"with\s+(\d+)\s+passengers?\s+and\s+(\d+)\s+crew", re.I)
BEHALF_RE = re.compile(r"on behalf of\s+([A-Z][\w\s\.&]+?)(?:\s+performing|,|\s+registration)", re.I)
ARTICLE_ID_RE = re.compile(r"article=([0-9a-f]{6,10})", re.I)
PAREN_COUNTRY_RE = re.compile(r"\(([A-Za-z\s\.]+)\)\s*$")


def parse_headline(h: str) -> dict:
    m = HEADLINE_RE.match((h or "").strip())
    if not m:
        return {}
    d = m.groupdict()
    d["date"] = nz.parse_loose_date(d["date"])
    return d


def parse_first_paragraph(p: Optional[str]) -> dict:
    out: dict = {}
    if not p:
        return out
    if m := REG_RE.search(p):
        out["reg"] = m.group(1)
    if m := FLIGHT_RE.search(p):
        out["flight"] = m.group(1)
    if m := ROUTE_RE.search(p):
        out["dep"], out["dest"] = m.group("dep").strip(), m.group("dest").strip()
    if m := POB_RE.search(p):
        out["occupants"] = int(m.group(1)) + int(m.group(2))
    if m := BEHALF_RE.search(p):
        out["marketing_carrier"] = m.group(1).strip()
    return out


def read_extracts(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        with open(path, encoding="utf-8") as fh:
            return pd.DataFrame([json.loads(x) for x in fh if x.strip()])
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def _country_of(place: Optional[str]) -> Optional[str]:
    if not place:
        return None
    m = PAREN_COUNTRY_RE.search(place)
    return nz.country_iso(m.group(1)) if m else None


def to_canonical(raw: pd.DataFrame, keep_text: bool = False) -> pd.DataFrame:
    rows = []
    for _, r in raw.iterrows():
        url = nz.clean_str(r.get("url"))
        aid = nz.clean_str(r.get("article_id")) or (ARTICLE_ID_RE.search(url).group(1) if url and ARTICLE_ID_RE.search(url) else None)
        h = parse_headline(nz.clean_str(r.get("headline")) or "")
        if not aid or not h:
            continue
        p = parse_first_paragraph(nz.clean_str(r.get("first_paragraph")))
        op_text = h.get("op")
        operator_icao = nz.operator_icao(op_text)
        flight = p.get("flight")
        # Codeshare / "on behalf of": the flight number belongs to the marketing carrier.
        flight_norm = nz.norm_flight_no(flight)
        dest_country = _country_of(p.get("dest"))
        dep_country = _country_of(p.get("dep"))
        place_country = nz.country_iso(h.get("place"))
        # Event country: explicit in headline place, else dep==dest country, else unknown.
        country = place_country or (dep_country if dep_country and dep_country == dest_country else None)
        reg_country = "US" if (p.get("reg") or "").upper().startswith("N") else None
        para = nz.clean_str(r.get("first_paragraph"))
        rows.append(dict(
            source="avherald",
            source_id=aid,
            event_date_local=h.get("date"),
            reg_raw=p.get("reg"),
            reg_norm=nz.norm_reg(p.get("reg"), reg_country),
            make_model_raw=h.get("type"),
            icao_type=h.get("type") if h.get("type") in refs.ICAO_TYPE_FAMILY else None,
            type_family=nz.type_family(h.get("type")),
            operator_raw=op_text,
            operator_icao=operator_icao,
            flight_no_norm=flight_norm,
            country_iso=country,
            location_raw=h.get("place"),
            occupants=p.get("occupants"),
            damage="UNK",
            category=nz.norm_category(h.get("cat")),
            phase=None,
            narrative=(" ".join(x for x in (h.get("cause"), para) if x) if keep_text else h.get("cause")),
            ntsb_no_refs=nz.extract_ntsb_numbers(para) if keep_text else [],
            docket_refs=[],
            url=url or f"https://avherald.com/h?article={aid}",
        ))
    return conform(pd.DataFrame(rows))
