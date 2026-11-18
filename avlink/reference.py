"""Reference tables: ICAO type designators → family, airlines, countries.

The seeds below cover the common airline fleets and GA types so the pipeline
works out of the box. For production, extend them from authoritative files:

    load_type_designators("icao_doc8643.csv")   # columns: designator, family[, model_regex]
    load_airlines("airlines.csv")                # columns: name, iata, icao  (e.g. OpenFlights airlines.dat)
    load_countries("countries.csv")              # columns: name, iso2

All loaders mutate the module-level tables in place, so call them once at startup
(the CLI does this when the paths are set in avlink.toml / env vars).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# Aircraft types
# --------------------------------------------------------------------------- #

_FAMILIES: dict[str, list[str]] = {
    "B737MAX": ["B37M", "B38M", "B39M", "B3XM"],
    "B737NG": ["B736", "B737", "B738", "B739"],
    "B737CL": ["B731", "B732", "B733", "B734", "B735"],
    "B747": ["B741", "B742", "B743", "B744", "B748", "B74S", "B74R"],
    "B757": ["B752", "B753"],
    "B767": ["B762", "B763", "B764"],
    "B777": ["B772", "B773", "B77L", "B77W", "B778", "B779"],
    "B787": ["B788", "B789", "B78X"],
    "B717": ["B712"],
    "MD80": ["MD81", "MD82", "MD83", "MD87", "MD88", "MD90"],
    "MD11": ["MD11"],
    "A220": ["BCS1", "BCS3"],
    "A320F": ["A318", "A319", "A320", "A321", "A19N", "A20N", "A21N"],
    "A300": ["A306", "A30B", "A310"],
    "A330": ["A332", "A333", "A337", "A338", "A339"],
    "A340": ["A342", "A343", "A345", "A346"],
    "A350": ["A359", "A35K"],
    "A380": ["A388"],
    "E-JET": ["E170", "E75L", "E75S", "E190", "E195", "E290", "E295"],
    "ERJ": ["E135", "E145", "E35L"],
    "CRJ": ["CRJ1", "CRJ2", "CRJ7", "CRJ9", "CRJX"],
    "DHC8": ["DH8A", "DH8B", "DH8C", "DH8D"],
    "ATR": ["AT43", "AT45", "AT72", "AT75", "AT76"],
    "SF34": ["SF34"], "B190": ["B190"], "C208": ["C208"], "PC12": ["PC12"],
    "C150": ["C150", "C152"], "C172": ["C172"], "C182": ["C182"], "C206": ["C206", "C210"],
    "PA28": ["P28A", "P28B", "P28R", "P28T"], "PA32": ["PA32", "P32R"], "PA46": ["PA46", "P46T"],
    "SR20": ["SR20"], "SR22": ["SR22", "S22T"], "BE36": ["BE33", "BE35", "BE36"],
    "R22": ["R22"], "R44": ["R44"], "R66": ["R66"], "AS50": ["AS50", "EC30"],
    "B06": ["B06"], "B407": ["B407"], "EC35": ["EC35"],
}
ICAO_TYPE_FAMILY: dict[str, str] = {d: fam for fam, ds in _FAMILIES.items() for d in ds}

# Free-text model → designator (checked before family rules). Order matters: specific first.
MODEL_TO_DESIGNATOR: list[tuple[re.Pattern, str]] = [(re.compile(p), d) for p, d in [
    (r"737[\s\-]?(?:MAX[\s\-]?7|7\s*MAX)", "B37M"), (r"737[\s\-]?(?:MAX[\s\-]?8|8\s*MAX|-8\b)", "B38M"),
    (r"737[\s\-]?(?:MAX[\s\-]?9|9\s*MAX|-9\b)", "B39M"), (r"737[\s\-]?(?:MAX[\s\-]?10|10\b)", "B3XM"),
    (r"737[\s\-]?6\d\d", "B736"), (r"737[\s\-]?7\d\d", "B737"), (r"737[\s\-]?8\d\d", "B738"),
    (r"737[\s\-]?9\d\d", "B739"), (r"787[\s\-]?8", "B788"), (r"787[\s\-]?9", "B789"),
    (r"787[\s\-]?10", "B78X"), (r"A?321\s*NEO|A21N", "A21N"), (r"A?320\s*NEO|A20N", "A20N"),
    (r"\bA321\b", "A321"), (r"\bA320\b", "A320"), (r"\bA319\b", "A319"),
]]

# Regex fallbacks over "MAKE MODEL" text (NTSB: 'BOEING 737-9', ASN: 'Boeing 737-9 MAX').
TYPE_FAMILY_RULES: list[tuple[re.Pattern, str]] = [(re.compile(p), f) for p, f in [
    (r"737[\s\-]?(?:MAX|7|8|9|10)\b(?!\d)|737-?[789]\b|737\s?MAX", "B737MAX"),
    (r"737[\s\-]?[6-9]\d\d", "B737NG"), (r"737[\s\-]?[1-5]\d\d", "B737CL"),
    (r"\b747\b|747-", "B747"), (r"\b757\b|757-", "B757"), (r"\b767\b|767-", "B767"),
    (r"\b777\b|777-", "B777"), (r"\b787\b|787-", "B787"), (r"\b717\b", "B717"),
    (r"\bMD-?8\d|DC-?9-?8\d|\bMD-?90", "MD80"), (r"\bMD-?11\b", "MD11"),
    (r"\bA-?22[01]|BD-?500|CS-?[13]00", "A220"), (r"\bA-?3(?:18|19|20|21)", "A320F"),
    (r"\bA-?300|\bA-?310", "A300"), (r"\bA-?330", "A330"), (r"\bA-?340", "A340"),
    (r"\bA-?350", "A350"), (r"\bA-?380", "A380"),
    (r"ERJ[\s\-]?1[79]0|EMB[\s\-]?1[79]\d|\bE-?1[79][05]\b|E-?JET", "E-JET"),
    (r"ERJ[\s\-]?1[34]5|EMB[\s\-]?1[34]5", "ERJ"),
    (r"\bCRJ|CL-?600-?2[BCDE]|REGIONAL JET", "CRJ"), (r"DHC-?8|DASH ?8|\bQ[1-4]00\b", "DHC8"),
    (r"\bATR[\s\-]?(?:42|72)", "ATR"), (r"\b208[AB]?\b.*CARAVAN|CESSNA.*\b208", "C208"),
    (r"PC-?12", "PC12"), (r"CESSNA.*\b15[02]\b", "C150"), (r"CESSNA.*\b172", "C172"),
    (r"CESSNA.*\b182", "C182"), (r"CESSNA.*\b(?:206|210)", "C206"),
    (r"PIPER.*PA-?28|CHEROKEE|WARRIOR|ARCHER", "PA28"), (r"PIPER.*PA-?32|SARATOGA", "PA32"),
    (r"PIPER.*PA-?46|MALIBU|MERIDIAN", "PA46"), (r"CIRRUS.*SR-?20", "SR20"),
    (r"CIRRUS.*SR-?22", "SR22"), (r"BEECH.*(?:BONANZA|\b3[356]\b)", "BE36"),
    (r"ROBINSON.*R-?22", "R22"), (r"ROBINSON.*R-?44", "R44"), (r"ROBINSON.*R-?66", "R66"),
]]

MAKE_ALIASES = {
    "BOEING": "BOEING", "AIRBUS": "AIRBUS", "EMBRAER": "EMBRAER", "BOMBARDIER": "BOMBARDIER",
    "CANADAIR": "BOMBARDIER", "DE HAVILLAND": "DEHAVILLAND", "DEHAVILLAND": "DEHAVILLAND",
    "CESSNA": "CESSNA", "TEXTRON": "CESSNA", "PIPER": "PIPER", "BEECH": "BEECH", "RAYTHEON": "BEECH",
    "HAWKER": "BEECH", "CIRRUS": "CIRRUS", "MOONEY": "MOONEY", "ROBINSON": "ROBINSON",
    "BELL": "BELL", "AIRBUS HELICOPTERS": "AIRBUSHEL", "EUROCOPTER": "AIRBUSHEL", "SIKORSKY": "SIKORSKY",
    "GRUMMAN": "GRUMMAN", "MAULE": "MAULE", "DIAMOND": "DIAMOND", "AERONCA": "AERONCA",
    "LUSCOMBE": "LUSCOMBE", "STINSON": "STINSON", "TAYLORCRAFT": "TAYLORCRAFT", "VANS": "VANS",
    "VAN'S": "VANS", "AMERICAN CHAMPION": "ACHAMPION", "CHAMPION": "ACHAMPION", "PILATUS": "PILATUS",
    "GULFSTREAM": "GULFSTREAM", "LEARJET": "LEARJET", "DASSAULT": "DASSAULT", "MCDONNELL": "MCDONNELL",
}

# --------------------------------------------------------------------------- #
# Airlines: normalized name → ICAO, and ICAO ↔ IATA
# --------------------------------------------------------------------------- #

_AIRLINES = [  # (normalized name, IATA, ICAO)
    ("UNITED", "UA", "UAL"), ("AMERICAN", "AA", "AAL"), ("DELTA", "DL", "DAL"),
    ("SOUTHWEST", "WN", "SWA"), ("ALASKA", "AS", "ASA"), ("JETBLUE", "B6", "JBU"),
    ("SPIRIT", "NK", "NKS"), ("FRONTIER", "F9", "FFT"), ("HAWAIIAN", "HA", "HAL"),
    ("ALLEGIANT", "G4", "AAY"), ("SUN COUNTRY", "SY", "SCX"), ("SKYWEST", "OO", "SKW"),
    ("REPUBLIC", "YX", "RPA"), ("ENVOY", "MQ", "ENY"), ("PSA", "OH", "JIA"),
    ("PIEDMONT", "PT", "PDT"), ("ENDEAVOR", "9E", "EDV"), ("HORIZON", "QX", "QXE"),
    ("MESA", "YV", "ASH"), ("AIR WISCONSIN", "ZW", "AWI"), ("FEDEX", "FX", "FDX"),
    ("FEDERAL EXPRESS", "FX", "FDX"), ("UPS", "5X", "UPS"), ("UNITED PARCEL SERVICE", "5X", "UPS"),
    ("ATLAS", "5Y", "GTI"), ("KALITTA", "K4", "CKS"), ("AIR CANADA", "AC", "ACA"),
    ("WESTJET", "WS", "WJA"), ("JAZZ", "QK", "JZA"), ("LUFTHANSA", "LH", "DLH"),
    ("BRITISH", "BA", "BAW"), ("AIR FRANCE", "AF", "AFR"), ("KLM", "KL", "KLM"),
    ("RYANAIR", "FR", "RYR"), ("MALTA AIR", "AL", "MAY"), ("EASYJET", "U2", "EZY"),
    ("WIZZ", "W6", "WZZ"), ("EMIRATES", "EK", "UAE"), ("QATAR", "QR", "QTR"),
    ("ETIHAD", "EY", "ETD"), ("TURKISH", "TK", "THY"), ("SINGAPORE", "SQ", "SIA"),
    ("CATHAY PACIFIC", "CX", "CPA"), ("QANTAS", "QF", "QFA"), ("ANA", "NH", "ANA"),
    ("ALL NIPPON", "NH", "ANA"), ("JAPAN", "JL", "JAL"), ("KOREAN", "KE", "KAL"),
    ("AIR INDIA", "AI", "AIC"), ("INDIGO", "6E", "IGO"), ("LATAM", "LA", "LAN"),
    ("AEROMEXICO", "AM", "AMX"), ("VOLARIS", "Y4", "VOI"), ("AVIANCA", "AV", "AVA"),
    ("COPA", "CM", "CMP"), ("IBERIA", "IB", "IBE"), ("SWISS", "LX", "SWR"),
    ("AUSTRIAN", "OS", "AUA"), ("SAS", "SK", "SAS"), ("SCANDINAVIAN", "SK", "SAS"),
    ("FINNAIR", "AY", "FIN"), ("AEGEAN", "A3", "AEE"), ("TAP", "TP", "TAP"),
    ("VIRGIN ATLANTIC", "VS", "VIR"), ("CHINA SOUTHERN", "CZ", "CSN"),
    ("CHINA EASTERN", "MU", "CES"), ("AIR CHINA", "CA", "CCA"),
]
AIRLINE_NAME_TO_ICAO: dict[str, str] = {n: icao for n, _, icao in _AIRLINES}
ICAO_TO_IATA: dict[str, str] = {icao: iata for _, iata, icao in _AIRLINES}
IATA_TO_ICAO: dict[str, str] = {iata: icao for _, iata, icao in _AIRLINES}
AIRLINE_ICAO_SET: set[str] = set(ICAO_TO_IATA)

# --------------------------------------------------------------------------- #
# Countries
# --------------------------------------------------------------------------- #

COUNTRY_TO_ISO: dict[str, str] = {
    "USA": "US", "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US", "U.S.A": "US", "US": "US",
    "CANADA": "CA", "MEXICO": "MX", "BRAZIL": "BR", "ARGENTINA": "AR", "CHILE": "CL", "COLOMBIA": "CO",
    "PERU": "PE", "UNITED KINGDOM": "GB", "UK": "GB", "ENGLAND": "GB", "SCOTLAND": "GB",
    "IRELAND": "IE", "FRANCE": "FR", "GERMANY": "DE", "SPAIN": "ES", "ITALY": "IT", "PORTUGAL": "PT",
    "NETHERLANDS": "NL", "BELGIUM": "BE", "SWITZERLAND": "CH", "AUSTRIA": "AT", "SWEDEN": "SE",
    "NORWAY": "NO", "DENMARK": "DK", "FINLAND": "FI", "POLAND": "PL", "GREECE": "GR", "TURKEY": "TR",
    "TÜRKIYE": "TR", "RUSSIA": "RU", "UKRAINE": "UA", "ISRAEL": "IL", "EGYPT": "EG",
    "SAUDI ARABIA": "SA", "UNITED ARAB EMIRATES": "AE", "UAE": "AE", "QATAR": "QA", "INDIA": "IN",
    "PAKISTAN": "PK", "NEPAL": "NP", "CHINA": "CN", "HONG KONG": "HK", "TAIWAN": "TW", "JAPAN": "JP",
    "SOUTH KOREA": "KR", "KOREA": "KR", "PHILIPPINES": "PH", "INDONESIA": "ID", "MALAYSIA": "MY",
    "SINGAPORE": "SG", "THAILAND": "TH", "VIETNAM": "VN", "AUSTRALIA": "AU", "NEW ZEALAND": "NZ",
    "SOUTH AFRICA": "ZA", "KENYA": "KE", "NIGERIA": "NG", "ETHIOPIA": "ET",
    "CAYMAN ISLANDS": "KY", "BAHAMAS": "BS", "PUERTO RICO": "PR", "GUAM": "GU",
}
ISO_SET: set[str] = set(COUNTRY_TO_ISO.values())

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
    "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "GM", "PR", "VI", "AO", "GU", "PO", "MP",
}

# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def _rows(path: str | Path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        yield from csv.DictReader(fh)


def load_type_designators(path: str | Path) -> int:
    n = 0
    for r in _rows(path):
        d = (r.get("designator") or "").strip().upper()
        fam = (r.get("family") or d).strip().upper()
        if d:
            ICAO_TYPE_FAMILY[d] = fam
            n += 1
        rx = (r.get("model_regex") or "").strip()
        if rx and d:
            MODEL_TO_DESIGNATOR.append((re.compile(rx, re.I), d))
    return n


def load_airlines(path: str | Path) -> int:
    n = 0
    for r in _rows(path):
        name = re.sub(r"[^A-Z0-9 ]", "", (r.get("name") or "").upper()).strip()
        iata, icao = (r.get("iata") or "").strip().upper(), (r.get("icao") or "").strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", icao):
            continue
        name = re.sub(r"\b(AIRLINES|AIRLINE|AIR LINES|AIRWAYS|INC|LTD)\b", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        if name:
            AIRLINE_NAME_TO_ICAO.setdefault(name, icao)
        AIRLINE_ICAO_SET.add(icao)
        if re.fullmatch(r"[A-Z0-9]{2}", iata):
            ICAO_TO_IATA.setdefault(icao, iata)
            IATA_TO_ICAO.setdefault(iata, icao)
        n += 1
    return n


def load_countries(path: str | Path) -> int:
    n = 0
    for r in _rows(path):
        name, iso = (r.get("name") or "").strip().upper(), (r.get("iso2") or "").strip().upper()
        if name and len(iso) == 2:
            COUNTRY_TO_ISO[name] = iso
            ISO_SET.add(iso)
            n += 1
    return n
