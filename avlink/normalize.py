"""Field normalizers shared by every source loader.

All functions are pure, null-safe, and vectorizable with Series.map / apply.
Reference tables (types, airlines, countries) are seeded here and can be
extended at runtime from CSVs via `avlink.reference`.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Iterable, Optional

import pandas as pd

from . import reference as ref

# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

def is_null(v) -> bool:
    if v is None:
        return True
    try:
        if v is pd.NA or v is pd.NaT:
            return True
    except Exception:  # pragma: no cover
        pass
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip() in ("", "UNK", "UNKNOWN", "N/A", "NA", "NONE GIVEN", "-"):
        return True
    return False


def clean_str(v) -> Optional[str]:
    if is_null(v):
        return None
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s or None


# --------------------------------------------------------------------------- #
# Identifiers found in free text
# --------------------------------------------------------------------------- #

# Region office + 2-digit FY + type code + 3-digit sequence (+ optional letter for multi-aircraft).
# Older numbers used other offices (e.g. MIA, LAX, FTW, NYC, ATL, CHI, DEN, SEA); include them.
NTSB_NO_RE = re.compile(
    r"\b(?:ANC|CEN|ERA|WPR|DCA|GAA|MIA|LAX|FTW|NYC|ATL|CHI|DEN|SEA|IAD|BOS|FAA)"
    r"\d{2}(?:FA|LA|MA|IA|CA|WA|GA|RA|SA|PA)\d{3}[A-Z]?\b"
)
DOCKET_RE = re.compile(r"(?:ProjectID|projectid|Mkey|mkey|NTSBNumber)=(\d{4,7})")
CAROL_MKEY_RE = re.compile(r"data\.ntsb\.gov/carol-(?:main-public|repgen)/[^\s\"']*?/(\d{5,7})(?:/|\b)")


def extract_ntsb_numbers(*texts: Optional[str]) -> list[str]:
    out: set[str] = set()
    for t in texts:
        if t:
            out.update(m.group(0).upper() for m in NTSB_NO_RE.finditer(str(t).upper()))
    return sorted(out)


def extract_docket_ids(*texts: Optional[str]) -> list[str]:
    out: set[str] = set()
    for t in texts:
        if not t:
            continue
        out.update(m.group(1) for m in DOCKET_RE.finditer(str(t)))
        out.update(m.group(1) for m in CAROL_MKEY_RE.finditer(str(t)))
    return sorted(out)


# --------------------------------------------------------------------------- #
# Registration / MSN
# --------------------------------------------------------------------------- #

_US_BARE_MARK = re.compile(r"^[1-9][0-9]{0,4}$|^[1-9][0-9]{0,3}[A-Z]$|^[1-9][0-9]{0,2}[A-Z]{2}$")


def norm_reg(raw, country_iso: Optional[str] = None, assume_us: bool = False) -> Optional[str]:
    """Normalize a registration mark.

    * upper-case, strip spaces / hyphens / dots
    * US marks: NTSB, FAA MASTER and ASIAS sometimes drop the leading 'N';
      re-add it when the event is in the US (or assume_us) and the bare mark
      matches the US N-number grammar.
    """
    s = clean_str(raw)
    if not s:
        return None
    s = re.sub(r"[\s\-\.]", "", s.upper())
    s = s.split("/")[0]  # "N123AB/N456CD" -> first
    if not re.fullmatch(r"[A-Z0-9]{2,10}", s):
        return None
    if (assume_us or (country_iso or "").upper() == "US") and _US_BARE_MARK.match(s):
        s = "N" + s
    return s


def norm_msn(raw) -> Optional[str]:
    s = clean_str(raw)
    if not s:
        return None
    s = re.sub(r"[\s\-\./]", "", s.upper())
    s = s.lstrip("0") or "0"
    return s if len(s) <= 20 else None


# --------------------------------------------------------------------------- #
# Dates, times, time zones
# --------------------------------------------------------------------------- #

# Offsets (hours) for the time-zone codes that appear in NTSB `ev_tmzn`.
TZ_OFFSETS = {
    "UTC": 0, "GMT": 0, "Z": 0, "UT": 0,
    "EST": -5, "EDT": -4, "CST": -6, "CDT": -5, "MST": -7, "MDT": -6,
    "PST": -8, "PDT": -7, "AKST": -9, "AKDT": -8, "YST": -9, "YDT": -8,
    "HST": -10, "HDT": -9, "HAST": -10, "AST": -4, "ADT": -3,
    "CHST": 10, "GST": 10, "SST": -11, "NST": -3.5, "NDT": -2.5,
}


def parse_hhmm(v) -> Optional[tuple[int, int]]:
    """NTSB ev_time is an int like 1435 or 35; ASN prints '14:35' or 'c. 14:35'."""
    if is_null(v):
        return None
    s = str(v).strip()
    m = re.search(r"(\d{1,2}):(\d{2})", s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
    else:
        digits = re.sub(r"\D", "", s.split(".")[0])
        if not digits:
            return None
        n = int(digits)
        h, mi = divmod(n, 100)
    if 0 <= h <= 23 and 0 <= mi <= 59:
        return h, mi
    return None


def to_utc(date_local, time_v, tz_code) -> Optional[datetime]:
    if is_null(date_local):
        return None
    hm = parse_hhmm(time_v)
    code = (clean_str(tz_code) or "").upper()
    if hm is None or code not in TZ_OFFSETS:
        return None
    d = pd.Timestamp(date_local).to_pydatetime().replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
    return d - timedelta(hours=TZ_OFFSETS[code])


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def parse_loose_date(s) -> Optional[pd.Timestamp]:
    """Parses 'Oct 25th 2026', 'Monday 25 October 2026', '2026-10-25', '10/25/2026'."""
    s = clean_str(s)
    if not s:
        return None
    m = re.search(r"([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", s)
    if m and m.group(1).lower() in _MONTHS:
        return pd.Timestamp(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
    m = re.search(r"(\d{1,2})\s+([A-Za-z]{3})[a-z]*\.?\s+(\d{4})", s)
    if m and m.group(2).lower() in _MONTHS:
        return pd.Timestamp(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
    ts = pd.to_datetime(s, errors="coerce")
    return None if pd.isna(ts) else pd.Timestamp(ts).normalize()


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #

def parse_coord(v, is_lat: bool) -> Optional[float]:
    """Accepts decimal degrees, or NTSB packed DMS strings ('362543N', '0942135W')."""
    if is_null(v):
        return None
    s = str(v).strip().upper()
    try:
        x = float(s)
        lim = 90 if is_lat else 180
        return x if -lim <= x <= lim else None
    except ValueError:
        pass
    m = re.fullmatch(r"(\d{2,3})(\d{2})(\d{2}(?:\.\d+)?)([NSEW])", s)
    if not m:
        m2 = re.fullmatch(r"(\d{2,3})(\d{2})([NSEW])", s)
        if not m2:
            return None
        deg, mins, sec, hemi = int(m2.group(1)), int(m2.group(2)), 0.0, m2.group(3)
    else:
        deg, mins, sec, hemi = int(m.group(1)), int(m.group(2)), float(m.group(3)), m.group(4)
    if mins >= 60 or sec >= 60:
        return None
    val = deg + mins / 60 + sec / 3600
    if hemi in "SW":
        val = -val
    lim = 90 if is_lat else 180
    return val if -lim <= val <= lim else None


def haversine_km(lat1, lon1, lat2, lon2) -> Optional[float]:
    if any(is_null(x) for x in (lat1, lon1, lat2, lon2)):
        return None
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# Damage, category, phase
# --------------------------------------------------------------------------- #

def norm_damage(v) -> str:
    s = (clean_str(v) or "").upper()
    if not s:
        return "UNK"
    if s in ("DEST", "SUBS", "MINR", "NONE", "UNK"):
        return s
    if "DESTROY" in s or "WRITTEN OFF" in s or "HULL LOSS" in s or "W/O" in s:
        return "DEST"
    if "SUBST" in s:
        return "SUBS"
    if "MINOR" in s:
        return "MINR"
    if s.startswith("NONE") or s == "NO DAMAGE":
        return "NONE"
    return "UNK"


DAMAGE_RANK = {"DEST": 4, "SUBS": 3, "MINR": 2, "NONE": 1, "UNK": 0}


def norm_category(v) -> Optional[str]:
    s = (clean_str(v) or "").lower()
    for k in ("crash", "accident", "incident", "news", "report"):
        if s.startswith(k) or s == k[:3]:
            return k
    if s in ("acc", "a"):
        return "accident"
    if s in ("inc", "i"):
        return "incident"
    return s or None


# --------------------------------------------------------------------------- #
# Aircraft type → ICAO designator / type family
# --------------------------------------------------------------------------- #

def icao_designator(text) -> Optional[str]:
    """Return an ICAO Doc 8643 designator if `text` IS one (e.g. 'B39M') or maps to one."""
    s = clean_str(text)
    if not s:
        return None
    u = s.upper().replace(" ", "")
    if u in ref.ICAO_TYPE_FAMILY:
        return u
    for pat, des in ref.MODEL_TO_DESIGNATOR:
        if pat.search(s.upper()):
            return des
    return None


def type_family(*texts) -> Optional[str]:
    """Coarse family for blocking & gating, e.g. B737MAX, B737NG, A320F, E-JET, C172.

    Tries designator → family, then regex rules over 'make model', then a
    generic 'MAKE|first-model-token' fallback so GA types still block well.
    """
    joined = " ".join(clean_str(t) or "" for t in texts).upper().strip()
    if not joined:
        return None
    for t in texts:
        d = icao_designator(t)
        if d and d in ref.ICAO_TYPE_FAMILY:
            return ref.ICAO_TYPE_FAMILY[d]
    for pat, fam in ref.TYPE_FAMILY_RULES:
        if pat.search(joined):
            return fam
    # Fallback: manufacturer stem + first model token (digits/letters), e.g. 'PIPER|PA28'
    make = None
    for stem, canon in ref.MAKE_ALIASES.items():
        if stem in joined:
            make = canon
            break
    tok = re.search(r"\b([A-Z]{0,3}-?\d{2,4}[A-Z]?)\b", joined)
    if make and tok:
        model = tok.group(1).replace("-", "")
        model = re.sub(r"[A-Z]$", "", model) if re.search(r"\d[A-Z]$", model) else model
        return f"{make}|{model}"
    return None


# --------------------------------------------------------------------------- #
# Operators & flight numbers
# --------------------------------------------------------------------------- #

def operator_icao(text) -> Optional[str]:
    s = clean_str(text)
    if not s:
        return None
    u = s.upper()
    if re.fullmatch(r"[A-Z]{3}", u) and u in ref.AIRLINE_ICAO_SET:
        return u
    key = re.sub(r"[^A-Z0-9 ]", "", u)
    key = re.sub(r"\b(INC|LLC|LTD|CORP|CORPORATION|CO|COMPANY|DBA|D B A|AIRLINES|AIRLINE|AIR LINES|AIRWAYS)\b", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    if key in ref.AIRLINE_NAME_TO_ICAO:
        return ref.AIRLINE_NAME_TO_ICAO[key]
    for name, code in ref.AIRLINE_NAME_TO_ICAO.items():
        if len(name) >= 4 and re.search(rf"\b{re.escape(name)}\b", key):
            return code
    return None


_FLT_RE = re.compile(r"\b([A-Z]{3}(?=[\s\-]?\d)|[A-Z][A-Z0-9]|[A-Z0-9][A-Z])[\s\-]?(\d{1,5})[A-Z]?\b")


def norm_flight_no(raw) -> Optional[str]:
    """'UA-1854', 'UAL1854', 'ua 01854' → 'UA1854' (ICAO prefixes mapped to IATA when known)."""
    s = clean_str(raw)
    if not s:
        return None
    m = _FLT_RE.search(s.upper())
    if not m:
        return None
    pfx, num = m.group(1), str(int(m.group(2)))
    if len(pfx) == 3:
        pfx = ref.ICAO_TO_IATA.get(pfx, pfx)
    return f"{pfx}{num}"


# --------------------------------------------------------------------------- #
# Countries
# --------------------------------------------------------------------------- #

def country_iso(text) -> Optional[str]:
    s = clean_str(text)
    if not s:
        return None
    u = s.upper().strip(" .,()")
    if u in ref.COUNTRY_TO_ISO:
        return ref.COUNTRY_TO_ISO[u]
    if len(u) == 2 and u.isalpha() and u in ref.ISO_SET:
        return u
    # search the last comma-separated segment first ('Houston, TX, United States of America')
    for seg in reversed([p.strip(" .()") for p in u.split(",")]):
        if seg in ref.COUNTRY_TO_ISO:
            return ref.COUNTRY_TO_ISO[seg]
        if seg in ref.US_STATES:
            return "US"
    for name, iso in sorted(ref.COUNTRY_TO_ISO.items(), key=lambda kv: -len(kv[0])):
        if len(name) > 3 and re.search(rf"\b{re.escape(name)}\b", u):
            return iso
    return None


def as_list(v: Iterable | None) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    try:
        return [str(x) for x in v if not is_null(x)]
    except TypeError:
        return []
