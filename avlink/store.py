"""DuckDB warehouse: upserts, history, enrichment."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from .schema import CANONICAL_COLUMNS, DDL, conform

log = logging.getLogger(__name__)


class Warehouse:
    def __init__(self, path: str | Path = "avlink.duckdb"):
        self.path = str(path)
        self.con = duckdb.connect(self.path)
        self.con.execute(DDL)

    def close(self):
        self.con.close()

    # ------------------------------------------------------------------ #
    def upsert_records(self, df: pd.DataFrame) -> int:
        """Insert-or-replace by (source, source_id); every snapshot also goes to history."""
        if df.empty:
            return 0
        df = conform(df).drop_duplicates(["source", "source_id"], keep="last")
        self.con.register("_incoming", df)
        cols = ", ".join(CANONICAL_COLUMNS)
        self.con.execute(f"INSERT INTO source_record_history SELECT {cols} FROM _incoming")
        self.con.execute("""
            DELETE FROM source_record USING _incoming
            WHERE source_record.source = _incoming.source AND source_record.source_id = _incoming.source_id
        """)
        self.con.execute(f"INSERT INTO source_record SELECT {cols} FROM _incoming")
        self.con.unregister("_incoming")
        log.info("upserted %d %s records", len(df), ",".join(df["source"].unique()))
        return len(df)

    def records(self, where: str = "true") -> pd.DataFrame:
        return self.con.execute(f"SELECT * FROM source_record WHERE {where}").df()

    # ------------------------------------------------------------------ #
    def load_faa_registry(self, reg: pd.DataFrame) -> None:
        self.con.execute("DELETE FROM faa_registry")
        self.con.register("_reg", reg)
        self.con.execute("""INSERT INTO faa_registry SELECT reg_norm, serial_no, mfr, model, type_family,
                            valid_from, valid_to, is_current FROM _reg""")
        self.con.unregister("_reg")

    def load_airports(self, ourairports_csv: str | Path) -> None:
        """OurAirports airports.csv (ident, iata_code, local_code, name, municipality, iso_country, lat/lon)."""
        self.con.execute("DELETE FROM airport")
        self.con.execute(f"""
            INSERT INTO airport
            SELECT CASE WHEN length(ident)=4 THEN ident END, nullif(iata_code,''), nullif(local_code,''),
                   name, municipality, iso_country, latitude_deg, longitude_deg
            FROM read_csv_auto('{ourairports_csv}', header=true, all_varchar=false)
            WHERE type NOT IN ('closed', 'heliport', 'balloonport')
        """)

    # ------------------------------------------------------------------ #
    def enrich(self) -> None:
        """Fill gaps from reference tables. Idempotent; only fills NULLs."""
        c = self.con
        # 1) FAA registry: US registration + date → MSN / type family (time-aware!)
        if c.execute("SELECT count(*) FROM faa_registry").fetchone()[0]:
            c.execute("""
                UPDATE source_record s SET
                    msn = coalesce(s.msn, f.serial_no),
                    type_family = coalesce(s.type_family, f.type_family)
                FROM (
                    SELECT s2.source, s2.source_id, any_value(r.serial_no) AS serial_no,
                           any_value(r.type_family) AS type_family
                    FROM source_record s2 JOIN faa_registry r
                      ON s2.reg_norm = r.reg_norm
                     AND coalesce(s2.event_date_local, current_date) BETWEEN r.valid_from AND r.valid_to
                    WHERE s2.reg_norm LIKE 'N%' AND (s2.msn IS NULL OR s2.type_family IS NULL)
                    GROUP BY 1, 2 HAVING count(DISTINCT r.serial_no) = 1
                ) f
                WHERE s.source = f.source AND s.source_id = f.source_id
            """)
        # 2) Airports: FAA LID / IATA → ICAO, and coordinates when missing
        if c.execute("SELECT count(*) FROM airport").fetchone()[0]:
            c.execute("""
                UPDATE source_record s SET airport_icao = a.icao
                FROM airport a
                WHERE s.source = 'ntsb' AND s.airport_icao IS NOT NULL AND a.icao IS NOT NULL
                  AND a.iso_country = 'US' AND a.local_code = regexp_replace(s.airport_icao, '^K', '')
                  AND NOT EXISTS (SELECT 1 FROM airport a2 WHERE a2.icao = s.airport_icao)
            """)
            c.execute("""
                UPDATE source_record s SET lat = a.lat, lon = a.lon
                FROM airport a
                WHERE s.lat IS NULL AND s.airport_icao = a.icao
            """)
            c.execute("""
                UPDATE source_record s SET country_iso = a.iso_country
                FROM airport a
                WHERE s.country_iso IS NULL AND s.airport_icao = a.icao
            """)
        # 3) AvHerald US registrations ⇒ treat as US for N-prefix logic already done at parse time.

    def stats(self) -> pd.DataFrame:
        return self.con.execute("""
            SELECT source, count(*) n,
                   round(avg((reg_norm IS NOT NULL)::INT), 3) AS pct_reg,
                   round(avg((type_family IS NOT NULL)::INT), 3) AS pct_type,
                   round(avg((flight_no_norm IS NOT NULL)::INT), 3) AS pct_flight,
                   round(avg((lat IS NOT NULL)::INT), 3) AS pct_latlon,
                   round(avg((len(ntsb_no_refs) > 0)::INT), 3) AS pct_ntsb_ref
            FROM source_record GROUP BY 1 ORDER BY 1
        """).df()
