"""Cluster accepted links into unified events (connected components) + QA flags."""
from __future__ import annotations

import hashlib
import logging

import pandas as pd

from .normalize import DAMAGE_RANK
from .store import Warehouse

log = logging.getLogger(__name__)

ACCEPTED = ("auto_accept", "accepted")


class UnionFind:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _event_id(members: list[tuple[str, str]], ntsb_ev_ids: list[str]) -> str:
    if len(ntsb_ev_ids) == 1:
        return f"ntsb:{ntsb_ev_ids[0]}"
    anchor = min(f"{s}:{i}" for s, i in members)
    return "evt:" + hashlib.sha1(anchor.encode()).hexdigest()[:16]


def run(wh: Warehouse, max_cluster_size: int = 12) -> pd.DataFrame:
    recs = wh.records()
    links = wh.con.execute(
        f"SELECT * FROM record_link WHERE status IN {ACCEPTED}").df()

    uf = UnionFind()
    for s, i in zip(recs["source"], recs["source_id"]):
        uf.find((s, i))
    for _, l in links.iterrows():
        uf.union((l.left_source, l.left_id), (l.right_source, l.right_id))

    recs["root"] = [uf.find((s, i)) for s, i in zip(recs["source"], recs["source_id"])]
    link_stats = {}
    for _, l in links.iterrows():
        root = uf.find((l.left_source, l.left_id))
        st = link_stats.setdefault(root, {"min_score": 1.0, "methods": set()})
        st["min_score"] = min(st["min_score"], float(l.score))
        st["methods"].add(l.method)

    events, members = [], []
    for root, g in recs.groupby("root", sort=False):
        mem = list(zip(g["source"], g["source_id"]))
        ev_ids = sorted(g.loc[g["source"] == "ntsb", "event_group_id"].dropna().unique().tolist())
        tfs = set(g["type_family"].dropna())
        regs = g["reg_norm"].dropna()
        dates = pd.to_datetime(g["event_date_local"]).dropna()
        flags = []
        if len(ev_ids) > 1:
            flags.append("multi_ntsb_event")
        if len({t for t in tfs if "|" not in t}) > 1:
            flags.append("type_family_conflict")
        if len(dates) and (dates.max() - dates.min()).days > 2:
            flags.append("date_spread_gt_2d")
        if len(mem) > max_cluster_size:
            flags.append("oversized_cluster")
        if len(g["source"].unique()) < len(mem) and not (len(ev_ids) == 1 and (g["source"] == "ntsb").sum() == len(mem)):
            flags.append("intra_source_duplicates")
        st = link_stats.get(root, {"min_score": None, "methods": set()})
        uid = _event_id(mem, ev_ids)
        worst = max(g["damage"].fillna("UNK"), key=lambda d: DAMAGE_RANK.get(d, 0))
        events.append(dict(
            unified_event_id=uid,
            canonical_date=dates.min().date() if len(dates) else None,
            canonical_reg=regs.mode().iloc[0] if len(regs) else None,
            canonical_type_family=g["type_family"].dropna().mode().iloc[0] if g["type_family"].notna().any() else None,
            canonical_operator=g["operator_icao"].dropna().mode().iloc[0] if g["operator_icao"].notna().any() else None,
            ntsb_ev_ids=ev_ids,
            n_sources=int(g["source"].nunique()),
            n_records=len(mem),
            max_fatalities=int(g["fatalities"].max()) if g["fatalities"].notna().any() else None,
            worst_damage=worst,
            min_link_score=st["min_score"],
            link_methods=sorted(st["methods"]),
            flags=flags,
        ))
        members += [dict(unified_event_id=uid, source=s, source_id=i) for s, i in mem]

    ev = pd.DataFrame(events)
    mem = pd.DataFrame(members)
    c = wh.con
    c.execute("DELETE FROM event")
    c.execute("DELETE FROM event_member")
    c.register("_ev", ev)
    c.register("_mem", mem)
    c.execute("INSERT INTO event SELECT * FROM _ev")
    c.execute("INSERT INTO event_member SELECT * FROM _mem")
    c.unregister("_ev")
    c.unregister("_mem")
    summary = c.execute("""
        SELECT n_sources, count(*) AS events, sum(n_records) AS records,
               sum((len(flags) > 0)::INT) AS flagged
        FROM event GROUP BY 1 ORDER BY 1
    """).df()
    log.info("clusters:\n%s", summary.to_string(index=False))
    return ev
