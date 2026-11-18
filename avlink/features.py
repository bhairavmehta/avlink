"""Export one feature row per unified event for downstream classification.

Linkage confidence is carried as features (n_sources, min_link_score, methods,
flags) so models can discount low-confidence fused records.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .store import Warehouse

FEATURE_SQL = """
WITH m AS (
    SELECT em.unified_event_id, s.*
    FROM event_member em JOIN source_record s USING (source, source_id)
)
SELECT
    e.unified_event_id, e.canonical_date, e.canonical_reg, e.canonical_type_family, e.canonical_operator,
    e.ntsb_ev_ids, e.n_sources, e.n_records, e.max_fatalities, e.worst_damage,
    coalesce(e.min_link_score, 1.0) AS min_link_score, e.link_methods, e.flags,
    bool_or(m.source = 'ntsb')     AS in_ntsb,
    bool_or(m.source = 'asn')      AS in_asn,
    bool_or(m.source = 'avherald') AS in_avherald,
    max(m.occupants)               AS max_occupants,
    any_value(m.country_iso) FILTER (WHERE m.country_iso IS NOT NULL)   AS country_iso,
    any_value(m.airport_icao) FILTER (WHERE m.airport_icao IS NOT NULL) AS airport_icao,
    any_value(m.lat) FILTER (WHERE m.lat IS NOT NULL) AS lat,
    any_value(m.lon) FILTER (WHERE m.lon IS NOT NULL) AS lon,
    any_value(m.phase) FILTER (WHERE m.phase IS NOT NULL) AS phase,
    list(DISTINCT m.category) FILTER (WHERE m.category IS NOT NULL) AS categories,
    any_value(m.category) FILTER (WHERE m.source = 'avherald') AS avherald_category,
    any_value(m.msn) FILTER (WHERE m.msn IS NOT NULL) AS msn,
    list(m.url) FILTER (WHERE m.url IS NOT NULL) AS urls,
    -- narratives kept separate per source so licence rules can be applied per column
    string_agg(m.narrative, ' ') FILTER (WHERE m.source = 'ntsb') AS ntsb_narrative,
    string_agg(m.narrative, ' ') FILTER (WHERE m.source = 'asn')  AS asn_narrative
FROM event e JOIN m USING (unified_event_id)
GROUP BY ALL
"""


def export(wh: Warehouse, out_path: str | Path) -> pd.DataFrame:
    df = wh.con.execute(FEATURE_SQL).df()
    out_path = Path(out_path)
    if out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)
    return df
