"""Deterministic linkage tiers, executed in DuckDB SQL.

T0  shared explicit identifier: NTSB number or docket ProjectID/mkey
T1  same normalized registration, |Δdate| ≤ window
T2  registration Damerau-Levenshtein ≤ 1 (typo/transposition), |Δdate| ≤ window, same type family
T3  same flight number, |Δdate| ≤ window, operator & type family not contradicting
SAME_EVENT  NTSB aircraft rows that share an ev_id (mid-airs, ground collisions)

Contradiction guards turn an otherwise-matching pair into `needs_review`
(e.g. same N-number but different type family → likely a re-registered mark).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pandas as pd

from .store import Warehouse

log = logging.getLogger(__name__)

RULE_VERSION = "rules-v1"


@dataclass
class DetConfig:
    date_window_days: int = 1
    t0_max_date_gap_days: int = 3        # explicit refs further apart than this → review
    min_reg_len_for_fuzzy: int = 4       # don't fuzzy-match very short marks
    same_source_dedupe: tuple[str, ...] = ("asn", "avherald")  # dedupe within these sources too


_PAIR_ORDER = "(l.source || chr(31) || l.source_id) < (r.source || chr(31) || r.source_id)"


def _same_source_clause(cfg: DetConfig) -> str:
    srcs = ", ".join(f"'{s}'" for s in cfg.same_source_dedupe) or "''"
    return f"(l.source <> r.source OR l.source IN ({srcs}))"


def candidate_features_sql(cfg: DetConfig) -> str:
    """Pairs within the date window that share at least one strong-ish key."""
    w = cfg.date_window_days
    return f"""
    WITH rec AS (SELECT * FROM source_record WHERE event_date_local IS NOT NULL)
    SELECT
        l.source AS left_source, l.source_id AS left_id, r.source AS right_source, r.source_id AS right_id,
        abs(date_diff('day', l.event_date_local, r.event_date_local)) AS d_days,
        l.reg_norm AS l_reg, r.reg_norm AS r_reg,
        CASE WHEN l.reg_norm IS NOT NULL AND r.reg_norm IS NOT NULL
             THEN damerau_levenshtein(l.reg_norm, r.reg_norm) END AS reg_dl,
        l.type_family AS l_tf, r.type_family AS r_tf,
        l.flight_no_norm AS l_flt, r.flight_no_norm AS r_flt,
        l.operator_icao AS l_op, r.operator_icao AS r_op,
        l.msn AS l_msn, r.msn AS r_msn,
        CASE WHEN l.lat IS NOT NULL AND r.lat IS NOT NULL THEN
            2 * 6371.0088 * asin(sqrt(
                pow(sin(radians(r.lat - l.lat) / 2), 2) +
                cos(radians(l.lat)) * cos(radians(r.lat)) * pow(sin(radians(r.lon - l.lon) / 2), 2)))
        END AS dist_km,
        l.country_iso AS l_cty, r.country_iso AS r_cty,
        l.fatalities AS l_fat, r.fatalities AS r_fat
    FROM rec l JOIN rec r
      ON r.event_date_local BETWEEN l.event_date_local - INTERVAL {w} DAY
                                AND l.event_date_local + INTERVAL {w} DAY
    WHERE {_PAIR_ORDER} AND {_same_source_clause(cfg)}
      AND NOT (l.source = 'ntsb' AND r.source = 'ntsb')
      AND (
            (l.reg_norm IS NOT NULL AND r.reg_norm IS NOT NULL
             AND abs(length(l.reg_norm) - length(r.reg_norm)) <= 1
             AND (l.reg_norm = r.reg_norm OR
                  (length(l.reg_norm) >= {cfg.min_reg_len_for_fuzzy}
                   AND left(l.reg_norm, 1) = left(r.reg_norm, 1))))
         OR (l.flight_no_norm IS NOT NULL AND l.flight_no_norm = r.flight_no_norm)
      )
    """


def tiers_sql(cfg: DetConfig) -> str:
    w = cfg.date_window_days
    return f"""
    WITH c AS ({candidate_features_sql(cfg)}),
    contra AS (
        SELECT *,
            (l_tf IS NOT NULL AND r_tf IS NOT NULL AND l_tf <> r_tf) AS tf_conflict,
            (l_msn IS NOT NULL AND r_msn IS NOT NULL AND l_msn <> r_msn) AS msn_conflict,
            (l_op IS NOT NULL AND r_op IS NOT NULL AND l_op <> r_op) AS op_conflict,
            (l_cty IS NOT NULL AND r_cty IS NOT NULL AND l_cty <> r_cty) AS cty_conflict,
            (dist_km IS NOT NULL AND dist_km > 300) AS far_apart
        FROM c
    ),
    tiered AS (
        SELECT *,
          CASE
            WHEN l_reg = r_reg AND d_days <= {w} THEN 'T1'
            WHEN reg_dl = 1 AND d_days <= {w} AND l_tf IS NOT NULL AND l_tf = r_tf
                 AND length(l_reg) >= {cfg.min_reg_len_for_fuzzy} THEN 'T2'
            WHEN l_flt IS NOT NULL AND l_flt = r_flt AND d_days <= {w}
                 AND NOT tf_conflict THEN 'T3'
          END AS method
        FROM contra
    )
    SELECT left_source, left_id, right_source, right_id, method,
        CASE method WHEN 'T1' THEN 0.99 WHEN 'T2' THEN 0.90 WHEN 'T3' THEN 0.90 END
          * CASE WHEN tf_conflict OR msn_conflict OR far_apart THEN 0.6 ELSE 1 END
          * CASE WHEN method = 'T3' AND op_conflict THEN 0.85 ELSE 1 END AS score,
        CASE WHEN tf_conflict OR msn_conflict OR far_apart OR (method = 'T3' AND op_conflict)
                  OR (method = 'T2' AND d_days > 0)
             THEN 'needs_review' ELSE 'auto_accept' END AS status,
        to_json({{'d_days': d_days, 'reg': [l_reg, r_reg], 'reg_dl': reg_dl, 'type_family': [l_tf, r_tf],
                  'flight': [l_flt, r_flt], 'operator': [l_op, r_op], 'msn': [l_msn, r_msn],
                  'dist_km': round(dist_km, 1), 'country': [l_cty, r_cty], 'fatalities': [l_fat, r_fat],
                  'conflicts': list_filter([CASE WHEN tf_conflict THEN 'type_family' END,
                                            CASE WHEN msn_conflict THEN 'msn' END,
                                            CASE WHEN op_conflict THEN 'operator' END,
                                            CASE WHEN far_apart THEN 'distance' END], lambda x: x IS NOT NULL)}}) AS evidence
    FROM tiered WHERE method IS NOT NULL
    """


def t0_sql(cfg: DetConfig) -> str:
    return f"""
    WITH refs AS (
        SELECT source, source_id, event_date_local, 'ntsb_no' AS kind, unnest(ntsb_no_refs) AS ref FROM source_record
        UNION ALL
        SELECT source, source_id, event_date_local, 'docket' AS kind, unnest(docket_refs) AS ref FROM source_record
    )
    SELECT l.source AS left_source, l.source_id AS left_id, r.source AS right_source, r.source_id AS right_id,
           'T0' AS method,
           CASE WHEN abs(date_diff('day', l.event_date_local, r.event_date_local)) > {cfg.t0_max_date_gap_days}
                THEN 0.7 ELSE 1.0 END AS score,
           CASE WHEN abs(date_diff('day', l.event_date_local, r.event_date_local)) > {cfg.t0_max_date_gap_days}
                THEN 'needs_review' ELSE 'auto_accept' END AS status,
           to_json({{'ref_kind': any_value(l.kind), 'refs': list(DISTINCT l.ref),
                     'd_days': abs(date_diff('day', any_value(l.event_date_local), any_value(r.event_date_local)))}}) AS evidence
    FROM refs l JOIN refs r ON l.kind = r.kind AND l.ref = r.ref
    WHERE {_PAIR_ORDER} AND (l.source <> r.source OR l.source <> 'ntsb')
    GROUP BY ALL
    """


SAME_EVENT_SQL = f"""
    SELECT l.source AS left_source, l.source_id AS left_id, r.source AS right_source, r.source_id AS right_id,
           'SAME_EVENT' AS method, 1.0 AS score, 'auto_accept' AS status,
           to_json({{'ev_id': l.event_group_id}}) AS evidence
    FROM source_record l JOIN source_record r
      ON l.source = 'ntsb' AND r.source = 'ntsb' AND l.event_group_id = r.event_group_id
    WHERE {_PAIR_ORDER}
"""

_RANK = "CASE method WHEN 'T0' THEN 0 WHEN 'SAME_EVENT' THEN 0 WHEN 'T1' THEN 1 WHEN 'T2' THEN 2 ELSE 3 END"


def run(wh: Warehouse, cfg: DetConfig | None = None) -> pd.DataFrame:
    """Compute all deterministic links and merge them into record_link.

    Human decisions (decided_by <> 'rules' and status in accepted/rejected) are never overwritten.
    """
    cfg = cfg or DetConfig()
    c = wh.con
    c.execute(f"""
        CREATE OR REPLACE TEMP TABLE _det AS
        SELECT * FROM (
            SELECT * FROM ({t0_sql(cfg)})
            UNION ALL BY NAME SELECT * FROM ({tiers_sql(cfg)})
            UNION ALL BY NAME SELECT * FROM ({SAME_EVENT_SQL})
        )
        QUALIFY row_number() OVER (PARTITION BY left_source, left_id, right_source, right_id
                                   ORDER BY {_RANK}, score DESC) = 1
    """)
    c.execute("DELETE FROM record_link WHERE decided_by = 'rules'")
    c.execute(f"""
        INSERT INTO record_link
        SELECT d.left_source, d.left_id, d.right_source, d.right_id, d.method, d.score, d.evidence::JSON,
               d.status, '{RULE_VERSION}', 'rules', now()
        FROM _det d
        WHERE NOT EXISTS (SELECT 1 FROM record_link k
                          WHERE k.left_source = d.left_source AND k.left_id = d.left_id
                            AND k.right_source = d.right_source AND k.right_id = d.right_id)
    """)
    summary = c.execute("""
        SELECT left_source || '↔' || right_source AS pair, method, status, count(*) AS n
        FROM _det GROUP BY ALL ORDER BY 1, 2, 3
    """).df()
    log.info("deterministic links:\n%s", summary.to_string(index=False))
    return summary
