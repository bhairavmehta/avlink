"""FAA Releasable Aircraft Database → time-aware registration table.

Download ReleasableAircraft.zip from
https://www.faa.gov/licenses_certificates/aircraft_certification/aircraft_registry/releasable_aircraft_download
and unzip. Uses MASTER.txt (current), DEREG.txt (deregistered history) and ACFTREF.txt
(make/model codes).

N-numbers are reassigned to different airframes over time, so every row has a
validity window and enrichment joins on (reg_norm, event_date BETWEEN valid_from AND valid_to).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import normalize as nz


def _read_faa(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1", low_memory=False)
    df.columns = [c.strip().upper().replace("-", " ").replace("_", " ") for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("UNNAMED")]]
    return df.apply(lambda s: s.str.strip())


def _date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.where(s.str.len() == 8), format="%Y%m%d", errors="coerce")


def load_registry(folder: str | Path) -> pd.DataFrame:
    folder = Path(folder)
    ref = _read_faa(folder / "ACFTREF.txt")
    ref = ref.rename(columns={"CODE": "MFR MDL CODE"})[["MFR MDL CODE", "MFR", "MODEL"]]

    frames = []
    master = _read_faa(folder / "MASTER.txt")
    m = master.merge(ref, on="MFR MDL CODE", how="left")
    frames.append(pd.DataFrame({
        "n_number": m["N NUMBER"], "serial_no": m["SERIAL NUMBER"], "mfr": m["MFR"], "model": m["MODEL"],
        "valid_from": _date(m.get("CERT ISSUE DATE", pd.Series(dtype=str))),
        "valid_to": pd.NaT, "is_current": True,
    }))

    dereg_path = folder / "DEREG.txt"
    if dereg_path.exists():
        d = _read_faa(dereg_path)
        code_col = "MFR MDL CODE" if "MFR MDL CODE" in d.columns else None
        if code_col:
            d = d.merge(ref, on=code_col, how="left")
        frames.append(pd.DataFrame({
            "n_number": d["N NUMBER"], "serial_no": d["SERIAL NUMBER"],
            "mfr": d.get("MFR", pd.Series(index=d.index, dtype=str)),
            "model": d.get("MODEL", pd.Series(index=d.index, dtype=str)),
            "valid_from": _date(d.get("CERT ISSUE DATE", pd.Series(index=d.index, dtype=str))),
            "valid_to": _date(d.get("CANCEL DATE", pd.Series(index=d.index, dtype=str))),
            "is_current": False,
        }))

    reg = pd.concat(frames, ignore_index=True)
    reg["reg_norm"] = reg["n_number"].map(lambda x: nz.norm_reg(x, "US", assume_us=True))
    reg["serial_no"] = reg["serial_no"].map(nz.norm_msn)
    reg["type_family"] = [nz.type_family(a, b) for a, b in zip(reg["mfr"], reg["model"])]
    reg["valid_from"] = reg["valid_from"].fillna(pd.Timestamp("1900-01-01"))
    reg["valid_to"] = reg["valid_to"].fillna(pd.Timestamp("2999-12-31"))
    return reg[["reg_norm", "serial_no", "mfr", "model", "type_family", "valid_from", "valid_to", "is_current"]] \
        .dropna(subset=["reg_norm"]).drop_duplicates()
