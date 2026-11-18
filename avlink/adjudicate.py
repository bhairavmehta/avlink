"""LLM 'select-among-candidates' adjudication for the grey zone.

For each anchor record that has `needs_review` links, the model sees the anchor
and ALL its candidates at once and must pick one candidate id or "none".
Each anchor is adjudicated `n_runs` times; only unanimous answers are applied,
disagreements stay `needs_review` for a human.

Plug any model in through `llm_fn(prompt: str) -> str`. `anthropic_llm_fn()` is a
ready-made implementation using the Anthropic SDK.

Licensing guard: AvHerald text is never sent unless `allow_avherald_text=True`
(only set that if your AvHerald licence covers third-party AI processing).
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from .store import Warehouse

log = logging.getLogger(__name__)

LLMFn = Callable[[str], str]

FIELDS = ["source", "source_id", "event_date_local", "reg_norm", "msn", "make_model_raw", "type_family",
          "operator_raw", "flight_no_norm", "location_raw", "country_iso", "airport_icao",
          "fatalities", "occupants", "damage", "phase", "category"]

PROMPT = """You are reconciling aviation safety records from different databases (NTSB, ASN, Aviation Herald).
Decide which CANDIDATE, if any, describes the SAME occurrence (same aircraft, same event) as the ANCHOR.

Rules:
- Registration typos (one transposed/wrong character) are common; aircraft type, date (±1 day for time zones),
  location and operator should still agree.
- An N-number can be reassigned to a different airframe over the years: different type family ⇒ not the same.
- Different events involving the same aircraft on adjacent days are possible (e.g. a diversion then a ferry flight).
- Prefer "none" over a guess.

ANCHOR:
{anchor}

CANDIDATES:
{candidates}

Reply with JSON only:
{{"match_id": "<candidate source_id or none>", "confidence": <0..1>, "evidence": ["..."], "conflicts": ["..."]}}"""


@dataclass
class AdjConfig:
    n_runs: int = 2
    max_candidates: int = 8
    narrative_chars: int = 600
    allow_avherald_text: bool = False
    min_confidence: float = 0.7
    limit_anchors: Optional[int] = None
    model_version: str = "llm-select-v1"


def _record_text(r: pd.Series, cfg: AdjConfig) -> str:
    d = {k: (None if pd.isna(r.get(k)) else str(r.get(k))) for k in FIELDS}
    nar = r.get("narrative")
    if isinstance(nar, str) and nar and (r["source"] != "avherald" or cfg.allow_avherald_text):
        d["narrative"] = nar[: cfg.narrative_chars]
    return json.dumps({k: v for k, v in d.items() if v not in (None, "", "UNK")}, ensure_ascii=False)


def _parse(resp: str) -> dict:
    m = re.search(r"\{.*\}", resp, re.S)
    if not m:
        return {"match_id": "none", "confidence": 0.0, "parse_error": resp[:200]}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"match_id": "none", "confidence": 0.0, "parse_error": resp[:200]}


def build_tasks(wh: Warehouse, cfg: AdjConfig) -> list[tuple[pd.Series, pd.DataFrame]]:
    links = wh.con.execute("""
        SELECT * FROM record_link WHERE status = 'needs_review'
    """).df()
    if links.empty:
        return []
    recs = wh.records().set_index(["source", "source_id"], drop=False)
    # Anchor = the side whose source sorts first (ntsb < asn < avherald by priority).
    prio = {"ntsb": 0, "asn": 1, "avherald": 2}
    rows = []
    for _, l in links.iterrows():
        a = (l.left_source, l.left_id)
        b = (l.right_source, l.right_id)
        if prio.get(b[0], 9) < prio.get(a[0], 9):
            a, b = b, a
        rows.append((a, b, l.score))
    df = pd.DataFrame(rows, columns=["anchor", "cand", "score"])
    tasks = []
    for anchor, grp in df.groupby("anchor"):
        if anchor not in recs.index:
            continue
        cands = [c for c in grp.sort_values("score", ascending=False)["cand"] if c in recs.index]
        cands = cands[: cfg.max_candidates]
        tasks.append((recs.loc[anchor], recs.loc[cands]))
        if cfg.limit_anchors and len(tasks) >= cfg.limit_anchors:
            break
    return tasks


def run(wh: Warehouse, llm_fn: LLMFn, cfg: AdjConfig | None = None) -> pd.DataFrame:
    cfg = cfg or AdjConfig()
    tasks = build_tasks(wh, cfg)
    log.info("adjudicating %d anchors", len(tasks))
    decisions = []
    for anchor, cands in tasks:
        prompt = PROMPT.format(
            anchor=_record_text(anchor, cfg),
            candidates="\n".join(f"- {_record_text(c, cfg)}" for _, c in cands.iterrows()),
        )
        answers = [_parse(llm_fn(prompt)) for _ in range(cfg.n_runs)]
        picks = Counter(str(a.get("match_id", "none")) for a in answers)
        pick, votes = picks.most_common(1)[0]
        unanimous = votes == cfg.n_runs
        conf = min(float(a.get("confidence", 0) or 0) for a in answers)
        valid_ids = set(cands["source_id"].astype(str))
        for _, c in cands.iterrows():
            cid = str(c["source_id"])
            if not unanimous or conf < cfg.min_confidence:
                status = "needs_review"
            elif pick == cid:
                status = "accepted"
            elif pick == "none" or pick in valid_ids:
                status = "rejected"
            else:
                status = "needs_review"
            decisions.append(dict(a_src=anchor["source"], a_id=str(anchor["source_id"]),
                                  c_src=c["source"], c_id=cid, status=status,
                                  evidence=json.dumps({"llm_runs": answers, "unanimous": unanimous})))
    if not decisions:
        return pd.DataFrame()
    dec = pd.DataFrame(decisions)
    wh.con.register("_dec", dec)
    wh.con.execute(f"""
        UPDATE record_link k SET status = d.status,
            evidence = json_merge_patch(k.evidence, d.evidence::JSON),
            decided_by = CASE WHEN d.status = 'needs_review' THEN k.decided_by ELSE 'llm' END,
            model_version = CASE WHEN d.status = 'needs_review' THEN k.model_version ELSE '{cfg.model_version}' END,
            decided_at = now()
        FROM _dec d
        WHERE (k.left_source = d.a_src AND k.left_id = d.a_id AND k.right_source = d.c_src AND k.right_id = d.c_id)
           OR (k.left_source = d.c_src AND k.left_id = d.c_id AND k.right_source = d.a_src AND k.right_id = d.a_id)
    """)
    wh.con.unregister("_dec")
    log.info("LLM decisions: %s", dec["status"].value_counts().to_dict())
    return dec


def anthropic_llm_fn(model: str = "claude-sonnet-5-5", max_tokens: int = 400) -> LLMFn:
    """Requires `pip install anthropic` and ANTHROPIC_API_KEY. Check your org's data policy first."""
    import anthropic
    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        msg = client.messages.create(model=model, max_tokens=max_tokens,
                                     messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return call


def heuristic_llm_fn(prompt: str) -> str:
    """Offline stand-in for tests/dry runs: picks the candidate with the most matching fields."""
    anchor = json.loads(re.search(r"ANCHOR:\n(\{.*?\})\n", prompt, re.S).group(1))
    cands = [json.loads(x[2:]) for x in re.findall(r"^- \{.*\}$", prompt, re.M)]
    keys = ["reg_norm", "type_family", "event_date_local", "flight_no_norm", "operator_raw", "fatalities"]
    best, best_s = "none", 0
    for c in cands:
        s = sum(1 for k in keys if k in anchor and anchor.get(k) == c.get(k))
        if anchor.get("type_family") and c.get("type_family") and anchor["type_family"] != c["type_family"]:
            s = -1
        if s > best_s:
            best, best_s = c["source_id"], s
    return json.dumps({"match_id": best if best_s >= 3 else "none", "confidence": 0.9 if best_s >= 3 else 0.8,
                       "evidence": [], "conflicts": []})
