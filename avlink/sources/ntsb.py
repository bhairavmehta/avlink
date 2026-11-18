"""NTSB ingest: avall.mdb (MS Access) bulk files and the NTSB Enterprise API.

Bulk files
----------
Download avall.zip (+ weekly upDDMON.zip change files) from https://data.ntsb.gov/avdata
and unzip. `mdbtools` must be installed (`apt-get install mdbtools` / `brew install mdbtools`).

    events, aircraft, narr = export_mdb_tables("avall.mdb")
    canon = to_canonical(events, aircraft, narr)

Change files contain the same tables; load them after avall and the warehouse
upsert (by source_id) applies the changes.

API
---
NTSB is moving the bulk download to its Enterprise API on 2027-04-05
(developer.ntsb.gov; header `Ocp-Apim-Subscription-Key`). The endpoint path and
response shape are configurable because they are versioned on NTSB's side —
confirm both on the developer portal before switching production ingest.
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import time
from datetime import date
from typing import Any, Iterable, Optional

import pandas as pd

from .. import normalize as nz
from ..schema import conform

log = logging.getLogger(__name__)

NTSB_TABLES = ("events", "aircraft", "narratives")


# --------------------------------------------------------------------------- #
# Bulk MDB
# --------------------------------------------------------------------------- #

def export_mdb_table(mdb_path: str, table: str) -> pd.DataFrame:
    if not shutil.which("mdb-export"):
        raise RuntimeError("mdb-export not found: install mdbtools (apt-get install mdbtools)")
    out = subprocess.run(
        ["mdb-export", "-D", "%Y-%m-%d %H:%M:%S", mdb_path, table],
        check=True, capture_output=True,
    ).stdout
    df = pd.read_csv(io.BytesIO(out), dtype=str, keep_default_na=False, na_values=[""], low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def export_mdb_tables(mdb_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ev = export_mdb_table(mdb_path, "events")
    ac = export_mdb_table(mdb_path, "aircraft")
    try:
        nr = export_mdb_table(mdb_path, "narratives")
    except subprocess.CalledProcessError:
        log.warning("narratives table not found in %s", mdb_path)
        nr = pd.DataFrame(columns=["ev_id", "aircraft_key", "narr_accp", "narr_cause"])
    return ev, ac, nr


def _g(row: pd.Series, *cols: str, default=None):
    for c in cols:
        if c in row.index and not nz.is_null(row[c]):
            return row[c]
    return default


def _sum_int(row: pd.Series, *cols: str) -> Optional[int]:
    vals = [pd.to_numeric(row[c], errors="coerce") for c in cols if c in row.index]
    vals = [v for v in vals if pd.notna(v)]
    return int(sum(vals)) if vals else None


def to_canonical(events: pd.DataFrame, aircraft: pd.DataFrame,
                 narratives: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One canonical row per (ev_id, aircraft) — i.e. per NTSB number.

    A mid-air is one ev_id with two ntsb_no; both rows share event_group_id so
    the clusterer keeps them in a single unified event.
    """
    ev = events.copy()
    ev.columns = [c.lower() for c in ev.columns]
    ac = aircraft.copy()
    ac.columns = [c.lower() for c in ac.columns]
    if "aircraft_key" not in ac.columns:
        ac["aircraft_key"] = "1"

    df = ac.merge(ev, on="ev_id", how="left", suffixes=("", "_ev"))
    if narratives is not None and len(narratives):
        nr = narratives.copy()
        nr.columns = [c.lower() for c in nr.columns]
        keep = [c for c in ("ev_id", "aircraft_key", "narr_accp", "narr_accf", "narr_cause", "narr_inc") if c in nr.columns]
        df = df.merge(nr[keep], on=[c for c in ("ev_id", "aircraft_key") if c in keep], how="left")

    rows = []
    for _, r in df.iterrows():
        country = nz.country_iso(_g(r, "ev_country")) or ("US" if _g(r, "ev_state") else None)
        ev_date = pd.to_datetime(_g(r, "ev_date"), errors="coerce")
        ev_date = None if pd.isna(ev_date) else ev_date.normalize()
        ntsb_no = nz.clean_str(_g(r, "ntsb_no"))
        make = nz.clean_str(_g(r, "acft_make", "make"))
        model = nz.clean_str(_g(r, "acft_model", "model"))
        lat = nz.parse_coord(_g(r, "dec_latitude"), True) or nz.parse_coord(_g(r, "latitude"), True)
        lon = nz.parse_coord(_g(r, "dec_longitude"), False) or nz.parse_coord(_g(r, "longitude"), False)
        narrative = " ".join(x for x in (nz.clean_str(_g(r, "narr_accp")), nz.clean_str(_g(r, "narr_accf")),
                                          nz.clean_str(_g(r, "narr_cause")), nz.clean_str(_g(r, "narr_inc"))) if x) or None
        apt = nz.clean_str(_g(r, "ev_nr_apt_id"))
        city, state = nz.clean_str(_g(r, "ev_city")), nz.clean_str(_g(r, "ev_state"))
        op = nz.clean_str(_g(r, "oper_name", "oper_dba", "operator"))
        rows.append(dict(
            source="ntsb",
            source_id=ntsb_no or f"{r['ev_id']}|{_g(r, 'aircraft_key', default='1')}",
            event_group_id=str(r["ev_id"]),
            event_date_local=ev_date,
            event_ts_utc=nz.to_utc(ev_date, _g(r, "ev_time"), _g(r, "ev_tmzn")),
            tz=nz.clean_str(_g(r, "ev_tmzn")),
            reg_raw=nz.clean_str(_g(r, "regis_no")),
            reg_norm=nz.norm_reg(_g(r, "regis_no"), country, assume_us=(country == "US")),
            msn=nz.norm_msn(_g(r, "acft_serial_no")),
            make_model_raw=" ".join(x for x in (make, model) if x) or None,
            icao_type=nz.icao_designator(model),
            type_family=nz.type_family(make, model),
            operator_raw=op,
            operator_icao=nz.operator_icao(_g(r, "oper_code")) or nz.operator_icao(op),
            flight_no_norm=nz.norm_flight_no(_g(r, "flight_no", "flt_no")),
            country_iso=country,
            location_raw=", ".join(x for x in (city, state, nz.clean_str(_g(r, "ev_country"))) if x) or None,
            airport_icao=_faa_lid_to_icao_guess(apt, country),
            dep_icao=_faa_lid_to_icao_guess(nz.clean_str(_g(r, "dprt_apt_id")), country),
            dest_icao=_faa_lid_to_icao_guess(nz.clean_str(_g(r, "dest_apt_id")), country),
            lat=lat, lon=lon,
            fatalities=_sum_int(r, "inj_tot_f") if "inj_tot_f" in r.index else _sum_int(r, "inj_f_grnd"),
            occupants=_sum_int(r, "inj_tot_f", "inj_tot_s", "inj_tot_m", "inj_tot_n"),
            damage=nz.norm_damage(_g(r, "damage")),
            phase=None,  # NTSB phase lives in Events_Sequence; join it in if you need it as a feature
            category=nz.norm_category({"ACC": "accident", "INC": "incident"}.get(
                str(_g(r, "ev_type", default="")).upper(), _g(r, "ev_type"))),
            narrative=narrative,
            ntsb_no_refs=[ntsb_no] if ntsb_no else [],
            docket_refs=[str(_g(r, "mkey"))] if _g(r, "mkey") else [],
            url=None,
        ))
    return conform(pd.DataFrame(rows))


def _faa_lid_to_icao_guess(lid: Optional[str], country: Optional[str]) -> Optional[str]:
    """NTSB stores FAA location IDs ('IAH', 'S43'). In CONUS, 3-letter alpha LIDs → 'K'+LID.

    This is a heuristic; `Warehouse.enrich_airports` replaces it with the
    OurAirports local_code mapping when an airport table is loaded.
    """
    if not lid:
        return None
    lid = lid.upper().strip()
    if len(lid) == 4 and lid.isalpha():
        return lid
    if country == "US" and len(lid) == 3 and lid.isalpha():
        return "K" + lid
    return None


# --------------------------------------------------------------------------- #
# Enterprise API
# --------------------------------------------------------------------------- #

class NtsbApiClient:
    """Minimal client for developer.ntsb.gov.

    Defaults target the aviation case-by-date-range endpoint NTSB recommends
    (GetCasesByDateRangeV2). Override `base_url` / `cases_path` / param names
    via env vars if the portal shows different values.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 cases_path: Optional[str] = None, timeout: int = 60,
                 max_calls_per_min: int = 900):
        import requests  # local import: optional dependency
        self._requests = requests
        self.api_key = api_key or os.environ["NTSB_API_KEY"]
        self.base_url = (base_url or os.getenv("NTSB_API_BASE", "https://api.ntsb.gov")).rstrip("/")
        self.cases_path = cases_path or os.getenv("NTSB_API_CASES_PATH", "/aviation/v2/GetCasesByDateRangeV2")
        self.timeout = timeout
        self.min_interval = 60.0 / max_calls_per_min
        self._last = 0.0

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(5):
            resp = self._requests.get(
                f"{self.base_url}{path}", params=params, timeout=self.timeout,
                headers={"Ocp-Apim-Subscription-Key": self.api_key, "Accept": "application/json"},
            )
            self._last = time.monotonic()
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(int(resp.headers.get("Retry-After", 2 ** attempt)))
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()

    def cases_by_date_range(self, start: date, end: date, chunk_days: int = 31) -> list[dict]:
        out: list[dict] = []
        cur = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        start_param = os.getenv("NTSB_API_START_PARAM", "StartDate")
        end_param = os.getenv("NTSB_API_END_PARAM", "EndDate")
        while cur <= end_ts:
            stop = min(cur + pd.Timedelta(days=chunk_days - 1), end_ts)
            payload = self._get(self.cases_path, {start_param: cur.strftime("%Y-%m-%d"),
                                                  end_param: stop.strftime("%Y-%m-%d")})
            items = payload if isinstance(payload, list) else (
                payload.get("value") or payload.get("Results") or payload.get("results") or payload.get("data") or [])
            out.extend(items)
            log.info("NTSB API %s..%s: %d cases", cur.date(), stop.date(), len(items))
            cur = stop + pd.Timedelta(days=1)
        return out


# Field aliases seen across CAROL CSV/JSON exports and API payloads.
_API_ALIASES = {
    "ntsb_no": ["NtsbNo", "NTSBNumber", "NtsbNumber", "ntsbNumber", "cm_ntsbNum"],
    "ev_id": ["EventID", "EvId", "ev_id", "eventId", "cm_eventId"],
    "mkey": ["Mkey", "MKey", "mkey", "cm_mkey", "ProjectID"],
    "ev_date": ["EventDate", "eventDate", "ev_date", "cm_eventDate"],
    "ev_time": ["EventTime", "ev_time"],
    "ev_tmzn": ["TimeZone", "ev_tmzn"],
    "ev_city": ["City", "cm_city", "ev_city"],
    "ev_state": ["State", "cm_state", "ev_state"],
    "ev_country": ["Country", "cm_country", "ev_country"],
    "latitude": ["Latitude", "cm_Latitude", "latitude"],
    "longitude": ["Longitude", "cm_Longitude", "longitude"],
    "regis_no": ["RegistrationNumber", "N#", "cm_regist", "regis_no"],
    "acft_make": ["Make", "cm_make", "acft_make"],
    "acft_model": ["Model", "cm_model", "acft_model"],
    "acft_serial_no": ["SerialNumber", "acft_serial_no"],
    "oper_name": ["Operator", "cm_operator", "oper_name"],
    "damage": ["AircraftDamage", "cm_damage", "damage"],
    "inj_tot_f": ["FatalInjuryCount", "cm_fatalInjuryCount", "inj_tot_f"],
    "inj_tot_s": ["SeriousInjuryCount", "cm_seriousInjuryCount", "inj_tot_s"],
    "inj_tot_m": ["MinorInjuryCount", "cm_minorInjuryCount", "inj_tot_m"],
    "ev_type": ["EventType", "cm_eventType", "ev_type"],
    "narr_cause": ["ProbableCause", "cm_probableCause", "narr_cause"],
    "narr_accp": ["AnalysisNarrative", "FactualNarrative", "narr_accp"],
    "dprt_apt_id": ["DepartureAirport", "dprt_apt_id"],
    "dest_apt_id": ["DestinationAirport", "dest_apt_id"],
    "ev_nr_apt_id": ["AirportID", "cm_airportID", "ev_nr_apt_id"],
}


def _flatten(item: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in item.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out.update(_flatten(v[0], f"{prefix}{k}."))  # first vehicle/aircraft
        else:
            out[f"{prefix}{k}"] = v
    return out


def api_cases_to_frames(cases: Iterable[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map API/CAROL JSON items to (events, aircraft, narratives) frames with avall column names."""
    flat = [_flatten(c) for c in cases]
    rows = []
    for f in flat:
        short = {k.split(".")[-1]: v for k, v in f.items()}
        row = {}
        for canon, aliases in _API_ALIASES.items():
            for a in aliases:
                if a in f and not nz.is_null(f[a]):
                    row[canon] = f[a]
                    break
                if a in short and not nz.is_null(short[a]):
                    row[canon] = short[a]
                    break
        row.setdefault("ev_id", row.get("mkey") or row.get("ntsb_no"))
        row["aircraft_key"] = "1"
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df, df
    ev_cols = [c for c in df.columns if c.startswith("ev_") or c in ("latitude", "longitude", "inj_tot_f", "inj_tot_s", "inj_tot_m", "mkey")]
    ac_cols = ["ev_id", "aircraft_key"] + [c for c in df.columns if c not in ev_cols and not c.startswith("narr_") and c not in ("ev_id", "aircraft_key")]
    nr_cols = ["ev_id", "aircraft_key"] + [c for c in df.columns if c.startswith("narr_")]
    events = df[list(dict.fromkeys(["ev_id"] + ev_cols))].drop_duplicates("ev_id")
    aircraft = df[list(dict.fromkeys(ac_cols + ["mkey"] if "mkey" in df.columns else ac_cols))]
    if "mkey" in aircraft.columns and "mkey" in events.columns:
        events = events.drop(columns=["mkey"])
    return events, aircraft, df[nr_cols]
