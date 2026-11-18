"""Synthetic NTSB / ASN / AvHerald data with known ground truth and realistic noise.

Noise injected:
  * ASN registration transposition typos (N5798T → N5789T)
  * ±1 day date shifts (time-zone / midnight effects)
  * NTSB regis_no without the leading 'N'
  * ASN sources carrying NTSB docket links or NTSB numbers (explicit refs) for a subset
  * mid-air collisions (one ev_id, two aircraft)
  * re-registered N-numbers on a different type years apart (must NOT link)
  * AvHerald-only incidents (no NTSB record), records with no registration
  * same flight number on consecutive days for different events (distractors)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd

TYPES = [  # (NTSB make, NTSB model, ASN type text, AvHerald designator, family, airline?)
    ("BOEING", "737-9", "Boeing 737-9 MAX", "B39M", True),
    ("BOEING", "737-800", "Boeing 737-800", "B738", True),
    ("AIRBUS", "A321-231", "Airbus A321-231", "A321", True),
    ("EMBRAER", "ERJ 170-200 LR", "Embraer ERJ 175LR", "E75L", True),
    ("BOEING", "787-9", "Boeing 787-9 Dreamliner", "B789", True),
    ("CESSNA", "172S", "Cessna 172S Skyhawk SP", None, False),
    ("PIPER", "PA-28-181", "Piper PA-28-181 Archer III", None, False),
    ("CIRRUS DESIGN CORP", "SR22", "Cirrus SR22", None, False),
    ("ROBINSON HELICOPTER", "R44 II", "Robinson R44 Raven II", None, False),
]
AIRLINES = [("United Airlines", "United", "UA"), ("American Airlines", "American", "AA"),
            ("Delta Air Lines", "Delta", "DL"), ("Alaska Airlines", "Alaska", "AS"),
            ("SkyWest Airlines", "Skywest", "OO")]
CITIES = [("Houston", "TX", 29.98, -95.34, "IAH"), ("Denver", "CO", 39.86, -104.67, "DEN"),
          ("Chicago", "IL", 41.98, -87.90, "ORD"), ("Seattle", "WA", 47.45, -122.31, "SEA"),
          ("Phoenix", "AZ", 33.43, -112.01, "PHX"), ("Atlanta", "GA", 33.64, -84.43, "ATL"),
          ("Boise", "ID", 43.56, -116.22, "BOI"), ("Fresno", "CA", 36.78, -119.72, "FAT")]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _ord(d: int) -> str:
    return f"{d}{'th' if 11 <= d % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')}"


def _dms(x: float, lat: bool) -> str:
    hemi = ("N" if x >= 0 else "S") if lat else ("E" if x >= 0 else "W")
    x = abs(x)
    d, m = int(x), int((x - int(x)) * 60)
    s = int(round(((x - d) * 60 - m) * 60))
    s = min(s, 59)
    return f"{d:02d}{m:02d}{s:02d}{hemi}" if lat else f"{d:03d}{m:02d}{s:02d}{hemi}"


@dataclass
class Synthetic:
    events: pd.DataFrame
    aircraft: pd.DataFrame
    narratives: pd.DataFrame
    asn: pd.DataFrame
    avherald: pd.DataFrame
    truth: dict[tuple[str, str], str] = field(default_factory=dict)  # (source, source_id) → true event id


def _transpose(reg: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(reg) - 1)
    return reg[:i] + reg[i + 1] + reg[i] + reg[i + 2:]


def make(n_events: int = 400, seed: int = 11) -> Synthetic:
    rng = random.Random(seed)
    ev_rows, ac_rows, nr_rows, asn_rows, avh_rows = [], [], [], [], []
    truth: dict[tuple[str, str], str] = {}
    used_regs: dict[str, int] = {}
    base = pd.Timestamp("2021-01-01")
    seq = {"ERA": 0, "CEN": 0, "WPR": 0, "DCA": 0}

    def new_reg() -> str:
        while True:
            r = f"N{rng.randint(100, 9999)}{rng.choice(['', 'A', 'T', 'XY', 'UA', 'DL'])}"
            if r not in used_regs:
                return r

    for k in range(n_events):
        gid = f"E{k:05d}"
        t = rng.choice(TYPES)
        make_, model, asn_type, avh_type, airline = t
        date = base + pd.Timedelta(days=rng.randint(0, 1400))
        city, st, lat, lon, apt = rng.choice(CITIES)
        lat += rng.uniform(-0.05, 0.05)
        lon += rng.uniform(-0.05, 0.05)
        reg = new_reg()
        used_regs[reg] = k
        msn = str(rng.randint(10000, 99999))
        fat = rng.choice([0, 0, 0, 0, 1, 2]) if not airline else 0
        dmg = rng.choice(["SUBS", "SUBS", "DEST", "MINR"]) if not airline else rng.choice(["MINR", "NONE", "SUBS"])
        op_full, op_short, iata = rng.choice(AIRLINES) if airline else ("Private", "Private", None)
        flt = f"{iata}-{rng.randint(10, 3999)}" if iata else None

        in_ntsb = (not airline) or rng.random() < 0.35      # most airline incidents never get an NTSB case
        in_asn = rng.random() < (0.9 if not airline else 0.6)
        in_avh = airline and rng.random() < 0.8
        midair = (not airline) and rng.random() < 0.02

        office = rng.choice(["ERA", "CEN", "WPR"]) if not airline else "DCA"
        ev_id = f"2021{k:06d}{rng.randint(10, 99)}"
        fy = str(date.year)[2:]
        nos = []
        if in_ntsb:
            seq[office] += 1
            ntsb_no = f"{office}{fy}{'LA' if not airline else 'IA'}{seq[office]:03d}"
            nos.append(ntsb_no)
            ev_rows.append(dict(ev_id=ev_id, ev_type="ACC" if not airline else "INC",
                                ev_date=date.strftime("%Y-%m-%d 00:00:00"), ev_time=str(rng.randint(600, 2200)),
                                ev_tmzn=rng.choice(["CST", "MST", "PST", "EST"]), ev_city=city.upper(), ev_state=st,
                                ev_country="USA", latitude=_dms(lat, True), longitude=_dms(lon, False),
                                ev_nr_apt_id=apt, inj_tot_f=str(fat * (2 if midair else 1)), inj_tot_s="0",
                                inj_tot_m="0", inj_tot_n=str(rng.randint(1, 4) if not airline else rng.randint(60, 180))))
            ac_rows.append(dict(ev_id=ev_id, aircraft_key="1", ntsb_no=ntsb_no,
                                regis_no=reg[1:] if rng.random() < 0.25 else reg,  # NTSB sometimes drops the N
                                acft_make=make_, acft_model=model, acft_serial_no=msn, damage=dmg,
                                oper_name=op_full if airline else None))
            nr_rows.append(dict(ev_id=ev_id, aircraft_key="1",
                                narr_cause=f"The pilot's failure to maintain control during {rng.choice(['landing', 'takeoff', 'cruise'])}."))
            truth[("ntsb", ntsb_no)] = gid
            if midair:
                reg2 = new_reg()
                used_regs[reg2] = k
                ntsb_no2 = ntsb_no + "B"
                nos[0] = ntsb_no + "A"
                ac_rows[-1]["ntsb_no"] = nos[0]
                truth.pop(("ntsb", ntsb_no))
                truth[("ntsb", nos[0])] = gid
                ac_rows.append(dict(ev_id=ev_id, aircraft_key="2", ntsb_no=ntsb_no2, regis_no=reg2,
                                    acft_make="CESSNA", acft_model="182T", acft_serial_no=str(rng.randint(10000, 99999)),
                                    damage="DEST", oper_name=None))
                truth[("ntsb", ntsb_no2)] = gid

        if in_asn:
            wid = str(200000 + k)
            asn_date = date + pd.Timedelta(days=rng.choice([0, 0, 0, 0, 1])) if rng.random() < 0.3 else date
            asn_reg = _transpose(reg, rng) if rng.random() < 0.06 else reg
            if rng.random() < 0.04:
                asn_reg = ""
            src_links = []
            if in_ntsb and rng.random() < 0.5:
                src_links.append(f"https://data.ntsb.gov/Docket?ProjectID={100000 + k}")
            report_no = nos[0] if (in_ntsb and rng.random() < 0.3) else ""
            asn_rows.append({
                "id": wid, "Date": f"{date.day_name()} {asn_date.day} {asn_date.strftime('%B %Y')}",
                "Time": f"{rng.randint(6, 22):02d}:{rng.randint(0, 59):02d}", "Type": asn_type,
                "Owner/operator": op_full, "Registration": asn_reg, "MSN": msn,
                "Fatalities": f"Fatalities: {fat} / Occupants: {rng.randint(1, 4)}",
                "Aircraft damage": {"DEST": "Destroyed, written off", "SUBS": "Substantial",
                                    "MINR": "Minor", "NONE": "None"}[dmg],
                "Category": "Accident" if not airline else "Incident",
                "Location": f"{city}, {st} - United States of America",
                "Phase": rng.choice(["Landing", "Take off", "En route", "Approach"]),
                "Departure airport": f"{city} Airport, {st} ({apt}/K{apt})",
                "Report number": report_no, "Narrative": f"The aircraft sustained damage near {city}. "
                + (f"Operating as flight {flt.replace('-', '')}." if flt and rng.random() < 0.5 else ""),
                "Sources": " ".join(src_links), "url": f"https://asn.flightsafety.org/wikibase/{wid}",
            })
            truth[("asn", f"wb:{wid}")] = gid
            if midair:
                asn_rows.append({**asn_rows[-1], "id": str(900000 + k), "Registration": reg2,
                                 "Type": "Cessna 182T", "MSN": "", "url": f"https://asn.flightsafety.org/wikibase/{900000 + k}"})
                truth[("asn", f"wb:{900000 + k}")] = gid
            # The docket ProjectID only links if NTSB carries the mkey: emulate CAROL by stashing it.
            if src_links and in_ntsb:
                for a in ac_rows:
                    if a["ev_id"] == ev_id:
                        a["mkey"] = str(100000 + k)

        if in_avh:
            aid = f"{0x50000000 + k:08x}"
            cat = rng.choice(["Incident", "Incident", "Accident"])
            avh_date = date + pd.Timedelta(days=rng.choice([0, 0, 0, -1])) if rng.random() < 0.2 else date
            show_reg = rng.random() < 0.9
            avh_rows.append(dict(
                url=f"https://avherald.com/h?article={aid}",
                headline=f"{cat}: {op_short} {avh_type} at {city} on {MONTHS[avh_date.month - 1]} {_ord(avh_date.day)} {avh_date.year}, "
                         f"{rng.choice(['bird strike', 'unsafe gear', 'smoke in cabin', 'engine shut down in flight'])}",
                first_paragraph=(f"A {op_short} {asn_type}" + (f", registration {reg}" if show_reg else "")
                                 + f" performing flight {flt} from Somewhere,{st} (USA) to {city},{st} (USA) "
                                 f"with {rng.randint(50, 170)} passengers and {rng.randint(4, 7)} crew, was on approach."),
            ))
            truth[("avherald", aid)] = gid
            # Distractor: same flight number next day, different aircraft, unrelated event in AvHerald
            if rng.random() < 0.05:
                did = f"{0x60000000 + k:08x}"
                d2 = date + pd.Timedelta(days=2)
                other = new_reg()
                avh_rows.append(dict(
                    url=f"https://avherald.com/h?article={did}",
                    headline=f"Incident: {op_short} {avh_type} near {city} on {MONTHS[d2.month - 1]} {_ord(d2.day)} {d2.year}, cracked windshield",
                    first_paragraph=f"A {op_short} {asn_type}, registration {other} performing flight {flt} from A,TX (USA) "
                                    f"to B,TX (USA) with 100 passengers and 5 crew, was climbing.",
                ))
                truth[("avherald", did)] = f"D{k:05d}"

        # Conflict guard: same N-number, next day, different type family (e.g. a data-entry error
        # or a reassigned mark) → must NOT auto-accept
        if rng.random() < 0.02 and in_ntsb:
            wid = str(700000 + k)
            asn_rows.append({"id": wid, "Date": (date + pd.Timedelta(days=1)).strftime("%d %b %Y"),
                             "Type": "Beechcraft A36 Bonanza", "Owner/operator": "Private", "Registration": reg,
                             "Fatalities": "Fatalities: 0 / Occupants: 1", "Aircraft damage": "Substantial",
                             "Category": "Accident", "Location": f"{city}, {st}",
                             "url": f"https://asn.flightsafety.org/wikibase/{wid}"})
            truth[("asn", f"wb:{wid}")] = f"R{k:05d}"

    return Synthetic(pd.DataFrame(ev_rows), pd.DataFrame(ac_rows), pd.DataFrame(nr_rows),
                     pd.DataFrame(asn_rows), pd.DataFrame(avh_rows), truth)
