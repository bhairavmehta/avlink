"""Wide outer-join view:  AvHerald ⟗ ASN ⟗ NTSB  over accepted links.

One row per matched combination; records with no partner still appear, with the
other sources' columns NULL. Join order:

  1. AvHerald FULL OUTER JOIN ASN       on an accepted avherald↔asn link
  2. (result) FULL OUTER JOIN NTSB      on the ASN row's NTSB link, or — if the
                                        row has no ASN-side NTSB link — the AvHerald row's NTSB link

A mid-air (one ASN/AvHerald record, two NTSB aircraft) yields two rows; that's intended.
`match_pattern` says which sources are present, e.g. 'AVH+ASN+NTSB', 'ASN+NTSB', 'AVH'.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .store import Warehouse

FIELDS = ["source_id", "event_date_local", "reg_norm", "type_family", "make_model_raw", "operator_icao",
          "operator_raw", "flight_no_norm", "location_raw", "country_iso", "fatalities", "occupants",
          "damage", "category", "url"]


def _cols(alias: str, prefix: str) -> str:
    return ",\n       ".join(f"{alias}.{f} AS {prefix}_{f}" for f in FIELDS)


WIDE_SQL = f"""
WITH acc AS (
    SELECT left_source, left_id, right_source, right_id, method, score
    FROM record_link
    WHERE status IN ('auto_accept', 'accepted') AND left_source <> right_source
),
sym AS (
    SELECT left_source AS a_src, left_id AS a_id, right_source AS b_src, right_id AS b_id, method, score FROM acc
    UNION ALL
    SELECT right_source, right_id, left_source, left_id, method, score FROM acc
),
avh_asn  AS (SELECT a_id AS avh_id, b_id AS asn_id,  method, score FROM sym WHERE a_src='avherald' AND b_src='asn'),
asn_ntsb AS (SELECT a_id AS asn_id, b_id AS ntsb_id, method, score FROM sym WHERE a_src='asn'      AND b_src='ntsb'),
avh_ntsb AS (SELECT a_id AS avh_id, b_id AS ntsb_id, method, score FROM sym WHERE a_src='avherald' AND b_src='ntsb'),
h AS (SELECT * FROM source_record WHERE source = 'avherald'),
a AS (SELECT * FROM source_record WHERE source = 'asn'),
n AS (SELECT * FROM source_record WHERE source = 'ntsb'),

-- step 1: AvHerald ⟗ ASN
ha AS (
    SELECT h.source_id AS avh_id, coalesce(a.source_id, l.asn_id) AS asn_id,
           l.method AS avh_asn_method, l.score AS avh_asn_score
    FROM h
    FULL OUTER JOIN avh_asn l ON l.avh_id = h.source_id
    FULL OUTER JOIN a ON a.source_id = l.asn_id
),
-- step 2: attach NTSB id (prefer the ASN-side link, fall back to the AvHerald-side link)
han AS (
    SELECT ha.*,
           coalesce(x.ntsb_id, y.ntsb_id) AS ntsb_id,
           coalesce(x.method, y.method)   AS ntsb_method,
           coalesce(x.score, y.score)     AS ntsb_score,
           CASE WHEN x.ntsb_id IS NOT NULL THEN 'asn' WHEN y.ntsb_id IS NOT NULL THEN 'avherald' END AS ntsb_linked_via
    FROM ha
    LEFT JOIN asn_ntsb x ON x.asn_id = ha.asn_id
    LEFT JOIN avh_ntsb y ON y.avh_id = ha.avh_id AND x.ntsb_id IS NULL
),
-- step 3: ⟗ NTSB (unlinked NTSB records come through as their own rows)
joined AS (
    SELECT han.avh_id, han.asn_id, coalesce(han.ntsb_id, n.source_id) AS ntsb_id,
           han.avh_asn_method, han.avh_asn_score, han.ntsb_method, han.ntsb_score, han.ntsb_linked_via
    FROM han FULL OUTER JOIN n ON n.source_id = han.ntsb_id
)
SELECT
    coalesce(em_h.unified_event_id, em_a.unified_event_id, em_n.unified_event_id) AS unified_event_id,
    concat_ws('+', CASE WHEN j.avh_id IS NOT NULL THEN 'AVH' END,
                   CASE WHEN j.asn_id IS NOT NULL THEN 'ASN' END,
                   CASE WHEN j.ntsb_id IS NOT NULL THEN 'NTSB' END) AS match_pattern,
    j.avh_asn_method, j.avh_asn_score, j.ntsb_method, j.ntsb_score, j.ntsb_linked_via,
    n.event_group_id AS ntsb_ev_id,
       {_cols('h', 'avh')},
       {_cols('a', 'asn')},
       {_cols('n', 'ntsb')}
FROM joined j
LEFT JOIN h ON h.source_id = j.avh_id
LEFT JOIN a ON a.source_id = j.asn_id
LEFT JOIN n ON n.source_id = j.ntsb_id
LEFT JOIN event_member em_h ON em_h.source = 'avherald' AND em_h.source_id = j.avh_id
LEFT JOIN event_member em_a ON em_a.source = 'asn'      AND em_a.source_id = j.asn_id
LEFT JOIN event_member em_n ON em_n.source = 'ntsb'     AND em_n.source_id = j.ntsb_id
ORDER BY coalesce(h.event_date_local, a.event_date_local, n.event_date_local), j.avh_id, j.asn_id, j.ntsb_id
"""


def build(wh: Warehouse) -> pd.DataFrame:
    return wh.con.execute(WIDE_SQL).df()


def export(wh: Warehouse, out_path: str | Path) -> pd.DataFrame:
    df = build(wh)
    out_path = Path(out_path)
    if out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    return df["match_pattern"].value_counts().rename_axis("match_pattern").reset_index(name="rows")
