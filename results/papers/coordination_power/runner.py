"""Independent reproduction of the Supreme Court results in "Echoes of power" (WWW 2012).

Danescu-Niculescu-Mizil, Lee, Pang & Kleinberg (2012), arXiv:1112.3670, Sections 3.2,
4 and 5.2-5.3 (Figures 4-5 and the Chief Justice contrast). Inputs are the public
ConvoKit ``supreme-corpus`` term files 2004-2006, kept outside git with checksums frozen
in ``configs/reproductions/coordination_power.json``. The measure is the existing
implementation in ``regesh.bonding.synchrony`` (Eq. 1-3 and Aggregated 1-3). Its
function-word lexicon is an authored open list, **not** the LIWC dictionary the paper
used, so absolute coordination levels are not comparable with the paper; directions
and significance are. No utterance text is written to the output directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import platform
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from regesh.bonding.synchrony import COORDINATION_MARKERS, LEXICON, CoordinationCounts, group_coordination

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - SciPy is optional
    scipy_stats = None

PAPER_URL = "https://arxiv.org/abs/1112.3670"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "reproductions" / "coordination_power.json"
AGGREGATES = ("aggregated_1", "aggregated_2", "aggregated_3")
QUANTITIES = AGGREGATES + tuple(COORDINATION_MARKERS)
STATUS_TYPES = ("J", "A")
# Group analyses: name -> (speaker type, target type). Entities are (speaker, case).
GROUPS = {
    "lawyers_to_justices": ("A", "J"), "justices_to_lawyers": ("J", "A"),
    "lawyers_to_chief": ("A", "J"), "lawyers_to_associates": ("A", "J"),
    "lawyers_to_unfavorable": ("A", "J"), "lawyers_to_favorable": ("A", "J"),
    "unfavorable_justices_to_lawyers": ("J", "A"), "favorable_justices_to_lawyers": ("J", "A"),
    "pair_lawyers_to_unfavorable": ("A", "J"), "pair_lawyers_to_favorable": ("A", "J"),
    "pair_unfavorable_justices_to_lawyers": ("J", "A"), "pair_favorable_justices_to_lawyers": ("J", "A"),
}
# Claim -> (hypothesised higher group, hypothesised lower group, paired sensitivity available).
CLAIMS = {
    "F4_lawyers_coordinate_more_than_justices": ("lawyers_to_justices", "justices_to_lawyers", False),
    "fine_status_chief_over_associates": ("lawyers_to_chief", "lawyers_to_associates", True),
    "F5a_lawyers_to_unfavorable_over_favorable": ("lawyers_to_unfavorable", "lawyers_to_favorable", True),
    "F5b_favorable_justices_over_unfavorable": ("favorable_justices_to_lawyers", "unfavorable_justices_to_lawyers", True),
}
PAIR_CLAIMS = {
    "F5a_pair_unit": ("pair_lawyers_to_unfavorable", "pair_lawyers_to_favorable"),
    "F5b_pair_unit": ("pair_favorable_justices_to_lawyers", "pair_unfavorable_justices_to_lawyers"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def verify_inputs(cache: Path, config: dict) -> dict[str, Path]:
    """Every listed input must exist and match its frozen SHA-256; nothing is fetched."""
    paths = {}
    for name, spec in config["inputs"].items():
        path = cache / name
        if not path.exists():
            raise FileNotFoundError(f"Missing input {path}; download the ConvoKit supreme-corpus term files first")
        digest = sha256(path)
        if digest != spec["sha256"] or ("bytes" in spec and path.stat().st_size != spec["bytes"]):
            raise ValueError(f"Input does not match the frozen checksum: {name}")
        paths[name] = path
    return paths


def read_term(zip_path: Path, term: int) -> dict:
    prefix = f"supreme-{term}/"
    with zipfile.ZipFile(zip_path) as archive:
        speakers = json.loads(archive.read(prefix + "speakers.json"))
        conversations = json.loads(archive.read(prefix + "conversations.json"))
        utterances = []
        with archive.open(prefix + "utterances.jsonl") as stream:
            for line in io.TextIOWrapper(stream, encoding="utf-8"):
                if line.strip():
                    utterances.append(json.loads(line))
    return {"term": term, "speakers": speakers, "conversations": conversations, "utterances": utterances}


def read_cases(path: Path) -> dict[str, dict]:
    cases = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                cases[record["id"]] = record
    return cases


def accepted_dockets(config: dict) -> tuple[set[str], dict[str, str]]:
    """Docket keys (case id without the 'YYYY_' prefix) selected, after optional aliasing."""
    aliases = config.get("case_id_aliases", {}) if config.get("case_id_matching", "aliased") == "aliased" else {}
    return {aliases.get(docket, docket) for docket in config["case_ids"]}, aliases


def select_conversations(terms: list[dict], config: dict) -> tuple[dict[str, dict], dict]:
    accepted, aliases = accepted_dockets(config)
    dockets_in_data: dict[str, set[str]] = {}
    selected = {}
    for data in terms:
        for conversation_id, meta in data["conversations"].items():
            docket = meta["case_id"][5:]
            dockets_in_data.setdefault(docket, set()).add(meta["case_id"])
            if docket in accepted:
                if conversation_id in selected:
                    raise ValueError(f"Conversation id repeated across term files: {conversation_id}")
                selected[conversation_id] = dict(meta, term=data["term"])
    exact = sorted(d for d in config["case_ids"] if d in dockets_in_data)
    via_alias = sorted(d for d in config["case_ids"] if d not in dockets_in_data and aliases.get(d) in dockets_in_data)
    missing = sorted(d for d in config["case_ids"] if d not in exact and d not in via_alias)
    cases = sorted({meta["case_id"] for meta in selected.values()})
    coverage = {"notebook_case_ids": len(config["case_ids"]), "matched_exactly": len(exact),
                "matched_via_alias": via_alias, "unmatched": missing,
                "case_id_matching": config.get("case_id_matching", "aliased"),
                "selected_conversations": len(selected), "selected_distinct_case_ids": len(cases),
                "cases_with_multiple_sessions": sorted(c for c in cases if sum(m["case_id"] == c for m in selected.values()) > 1),
                "conversations_in_term_files": sum(len(d["conversations"]) for d in terms),
                "selected_by_term": dict(sorted(Counter(m["term"] for m in selected.values()).items()))}
    return selected, coverage


def favorability(meta: dict, lawyer: str, justice: str) -> str:
    """'favorable'/'unfavorable' when both sides are 0/1 and comparable; otherwise the reason."""
    side = ((meta.get("advocates") or {}).get(lawyer) or {}).get("side")
    vote = (meta.get("votes_side") or {}).get(justice)
    if side not in (0, 1) or isinstance(side, bool):
        return "lawyer_side_undefined"
    if vote not in (0, 1) or isinstance(vote, bool):
        return "justice_vote_undefined"
    return "favorable" if int(side) == int(vote) else "unfavorable"


def collect_counts(terms: list[dict], selected: dict[str, dict], config: dict) -> tuple[dict, dict]:
    """Accumulate CoordinationCounts per (analysis, entity) over immediate-reply exchanges."""
    chiefs = set(config["chief_justice_ids"])
    counts: dict[str, dict[tuple, CoordinationCounts]] = {name: {} for name in GROUPS}
    dropped = Counter()
    kept = Counter()
    favor = Counter()
    speakers_seen: dict[str, set] = {"J": set(), "A": set()}
    entities_seen: dict[str, set] = {"J": set(), "A": set()}
    chief_terms: dict[str, set] = {chief: set() for chief in chiefs}
    utterances_selected = 0

    def add(name, key, u1_text, u2_text):
        counts[name].setdefault(key, CoordinationCounts(COORDINATION_MARKERS)).add(u1_text, u2_text)

    for data in terms:
        speaker_types = {sid: info.get("type") for sid, info in data["speakers"].items()}
        by_id = {u["id"]: u for u in data["utterances"] if u["conversation_id"] in selected}
        for u2 in by_id.values():
            utterances_selected += 1
            meta = selected[u2["conversation_id"]]
            if u2["meta"].get("case_id") != meta["case_id"]:
                raise ValueError(f"Utterance case id disagrees with its conversation: {u2['id']}")
            if u2["speaker"] in chief_terms:
                chief_terms[u2["speaker"]].add(data["term"])
            if u2.get("reply_to") is None:
                dropped["no_reply_to"] += 1
                continue
            u1 = by_id.get(u2["reply_to"])
            if u1 is None:
                dropped["reply_target_missing_or_outside_selection"] += 1
                continue
            if u1["conversation_id"] != u2["conversation_id"]:
                dropped["reply_across_conversations"] += 1
                continue
            if u1["speaker"] == u2["speaker"]:
                dropped["same_speaker"] += 1
                continue
            if not u1["text"].strip() or not u2["text"].strip():
                dropped["empty_text"] += 1
                continue
            t_type = speaker_types.get(u1["speaker"]) or u1["meta"].get("speaker_type")
            s_type = speaker_types.get(u2["speaker"]) or u2["meta"].get("speaker_type")
            if t_type not in STATUS_TYPES or s_type not in STATUS_TYPES:
                dropped["unknown_or_inaudible_speaker"] += 1
                continue
            if t_type == s_type:
                dropped[f"same_type_{s_type}_to_{t_type}"] += 1
                continue
            case = meta["case_id"]
            speaker, target = u2["speaker"], u1["speaker"]
            kept[f"{t_type}_target_{s_type}_speaker"] += 1
            speakers_seen[s_type].add(speaker)
            speakers_seen[t_type].add(target)
            entities_seen[s_type].add((speaker, case))
            entities_seen[t_type].add((target, case))
            entity, pair = (speaker, case), (speaker, target, case)
            if s_type == "A":
                add("lawyers_to_justices", entity, u1["text"], u2["text"])
                add("lawyers_to_chief" if target in chiefs else "lawyers_to_associates", entity, u1["text"], u2["text"])
                status = favorability(meta, speaker, target)
                favor[f"lawyer_speaker:{status}"] += 1
                if status in ("favorable", "unfavorable"):
                    add(f"lawyers_to_{status}", entity, u1["text"], u2["text"])
                    add(f"pair_lawyers_to_{status}", pair, u1["text"], u2["text"])
            else:
                add("justices_to_lawyers", entity, u1["text"], u2["text"])
                status = favorability(meta, target, speaker)
                favor[f"justice_speaker:{status}"] += 1
                if status in ("favorable", "unfavorable"):
                    add(f"{status}_justices_to_lawyers", entity, u1["text"], u2["text"])
                    add(f"pair_{status}_justices_to_lawyers", pair, u1["text"], u2["text"])
    data_summary = {"utterances_in_selected_conversations": utterances_selected,
                    "exchanges_kept": dict(sorted(kept.items())), "exchanges_kept_total": sum(kept.values()),
                    "dropped": dict(sorted(dropped.items())),
                    "favorability_exchanges": dict(sorted(favor.items())),
                    "distinct_justices": len(speakers_seen["J"]), "distinct_lawyers": len(speakers_seen["A"]),
                    "justice_entities": len(entities_seen["J"]), "lawyer_entities": len(entities_seen["A"]),
                    "justice_ids": sorted(speakers_seen["J"]),
                    "chief_justice_terms": {chief: sorted(terms_) for chief, terms_ in chief_terms.items()},
                    "lawyers_in_one_case_fraction": _one_case_fraction(entities_seen["A"]),
                    "max_cases_per_lawyer": max(Counter(s for s, _ in entities_seen["A"]).values(), default=0)}
    return counts, data_summary


def _one_case_fraction(entities) -> float | None:
    per_lawyer = Counter(speaker for speaker, _ in entities)
    return (sum(v == 1 for v in per_lawyer.values()) / len(per_lawyer)) if per_lawyer else None


def thresholded_summary(counts: CoordinationCounts, thresholds: dict) -> dict:
    """Eq. (1) per marker, undefined below ConvoKit-style marker-usage thresholds."""
    target_min = max(int(thresholds["target_min_marker_utterances"]), 1)
    speaker_min = int(thresholds["speaker_min_marker_utterances"])
    exchange_min = max(int(thresholds.get("min_exchanges", 1)), 1)
    by_marker = {}
    for marker in counts.markers:
        value = counts.coordination(marker)
        if counts.a[marker] < target_min or counts.b[marker] < speaker_min or counts.n < exchange_min:
            value = None
        by_marker[marker] = value
    return {"exchanges": counts.n, "by_marker": by_marker}


def evaluate_group(entities: dict[tuple, CoordinationCounts], thresholds: dict) -> tuple[dict, list[dict]]:
    """Group aggregation via synchrony.group_coordination plus matching per-entity values."""
    keys = sorted(entities)
    summaries = [thresholded_summary(entities[key], thresholds) for key in keys]
    group = group_coordination(summaries, COORDINATION_MARKERS)
    records = []
    for key, summary in zip(keys, summaries):
        defined = {m: v for m, v in summary["by_marker"].items() if v is not None}
        agg1 = agg2 = agg3 = None
        if defined:
            agg3 = sum(defined.values()) / len(defined)
            if len(defined) == len(COORDINATION_MARKERS):
                agg1 = agg3
            smoothed = [defined.get(m, group["by_marker"][m]) for m in COORDINATION_MARKERS]
            if all(v is not None for v in smoothed):
                agg2 = sum(smoothed) / len(smoothed)
        records.append({"key": key, "exchanges": summary["exchanges"], "by_marker": summary["by_marker"],
                        "aggregated_1": agg1, "aggregated_2": agg2, "aggregated_3": agg3})
    for name in AGGREGATES:  # the per-entity values must average to the library's group values
        values = [r[name] for r in records if r[name] is not None]
        mine = (sum(values) / len(values)) if values else None
        if (mine is None) != (group[name] is None) or (mine is not None and abs(mine - group[name]) > 1e-9):
            raise AssertionError(f"Per-entity {name} disagrees with group_coordination")
    return group, records


def vectors(records: list[dict]) -> dict[str, np.ndarray]:
    out = {name: np.asarray([r[name] for r in records if r[name] is not None], float) for name in AGGREGATES}
    for marker in COORDINATION_MARKERS:
        out[marker] = np.asarray([r["by_marker"][marker] for r in records if r["by_marker"][marker] is not None], float)
    return out


def _two_sided_p(t: float, df: float) -> tuple[float, str]:
    if scipy_stats is not None:
        return float(2 * scipy_stats.t.sf(abs(t), df)), "scipy.stats.t"
    return float(math.erfc(abs(t) / math.sqrt(2))), "normal approximation"


def welch(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return {"t": None, "df": None, "p": None, "method": "insufficient entities"}
    v1, v2 = float(x.var(ddof=1)), float(y.var(ddof=1))
    se2 = v1 / n1 + v2 / n2
    if se2 == 0:
        return {"t": None, "df": None, "p": None, "method": "zero variance"}
    t = (float(x.mean()) - float(y.mean())) / math.sqrt(se2)
    df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    p, method = _two_sided_p(t, df)
    return {"t": t, "df": df, "p": p, "method": method}


def student(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return {"t": None, "df": None, "p": None}
    pooled = ((n1 - 1) * float(x.var(ddof=1)) + (n2 - 1) * float(y.var(ddof=1))) / (n1 + n2 - 2)
    if pooled == 0:
        return {"t": None, "df": n1 + n2 - 2, "p": None}
    t = (float(x.mean()) - float(y.mean())) / math.sqrt(pooled * (1 / n1 + 1 / n2))
    p, _ = _two_sided_p(t, n1 + n2 - 2)
    return {"t": t, "df": n1 + n2 - 2, "p": p}


def paired_t(differences) -> dict:
    d = np.asarray(differences, float)
    n = len(d)
    if n < 2:
        return {"n_pairs": n, "t": None, "df": None, "p": None}
    sd = float(d.std(ddof=1))
    if sd == 0:
        return {"n_pairs": n, "mean_difference_pct": 100 * float(d.mean()), "t": None, "df": n - 1, "p": None}
    t = float(d.mean()) / (sd / math.sqrt(n))
    p, _ = _two_sided_p(t, n - 1)
    return {"n_pairs": n, "mean_difference_pct": 100 * float(d.mean()), "t": t, "df": n - 1, "p": p}


def bootstrap_difference(x, y, rng: np.random.Generator, samples: int, chunk: int = 500) -> list[float]:
    """Percentile 95% interval of mean(x)-mean(y), resampling entities within each group."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    diffs = np.empty(samples)
    done = 0
    while done < samples:
        k = min(chunk, samples - done)
        mx = x[rng.integers(0, len(x), size=(k, len(x)))].mean(axis=1)
        my = y[rng.integers(0, len(y), size=(k, len(y)))].mean(axis=1)
        diffs[done:done + k] = mx - my
        done += k
    return [float(v) for v in np.quantile(diffs, [.025, .975])]


def compare_groups(high: dict, low: dict, rng, samples: int, *, bootstrap_markers: bool, published: dict | None) -> dict:
    """Hypothesis high > low, tested per aggregate and per marker across entities."""
    out = {}
    for quantity in QUANTITIES:
        x, y = high["vectors"][quantity], low["vectors"][quantity]
        row = {"n": [int(len(x)), int(len(y))],
               "mean_pct": [100 * float(x.mean()) if len(x) else None, 100 * float(y.mean()) if len(y) else None]}
        row["difference_pct"] = (row["mean_pct"][0] - row["mean_pct"][1]) if None not in row["mean_pct"] else None
        row["welch"] = welch(x, y)
        row["student"] = student(x, y)
        if len(x) >= 2 and len(y) >= 2 and (bootstrap_markers or quantity in AGGREGATES):
            low_ci, high_ci = bootstrap_difference(x, y, rng, samples)
            row["bootstrap_95_ci_pct"] = [100 * low_ci, 100 * high_ci]
        row["direction_reproduced"] = (row["difference_pct"] > 0) if row["difference_pct"] is not None else None
        p = row["welch"]["p"]
        row["significant_0_05"] = (p < .05) if p is not None else None
        row["published_stars"] = (published or {}).get(quantity)
        out[quantity] = row
    return out


def paired_comparison(high_records: list[dict], low_records: list[dict]) -> dict:
    """Entities present in both groups: within-entity differences (sensitivity, not primary)."""
    low_by_key = {r["key"]: r for r in low_records}
    out = {}
    for quantity in QUANTITIES:
        differences = []
        for record in high_records:
            other = low_by_key.get(record["key"])
            if other is None:
                continue
            a = record[quantity] if quantity in AGGREGATES else record["by_marker"][quantity]
            b = other[quantity] if quantity in AGGREGATES else other["by_marker"][quantity]
            if a is not None and b is not None:
                differences.append(a - b)
        out[quantity] = paired_t(differences)
    return out


def percent(group: dict) -> dict:
    return {"speakers": group["speakers"], "speakers_complete": group["speakers_complete"],
            "by_marker_pct": {m: (100 * v if v is not None else None) for m, v in group["by_marker"].items()},
            **{name + "_pct": (100 * group[name] if group[name] is not None else None) for name in AGGREGATES}}


def analyse(counts: dict, thresholds: dict, rng, samples: int, config: dict, *, full: bool) -> tuple[dict, dict]:
    evaluated = {}
    for name, entities in counts.items():
        group, records = evaluate_group(entities, thresholds)
        evaluated[name] = {"group": group, "records": records, "vectors": vectors(records)}
    published = config.get("published", {}).get("claims", {})
    claims = {}
    for claim, (high, low, paired) in CLAIMS.items():
        claims[claim] = {"higher_group": high, "lower_group": low,
                         "entities": {high: evaluated[high]["group"]["speakers"], low: evaluated[low]["group"]["speakers"]},
                         "tests": compare_groups(evaluated[high], evaluated[low], rng, samples,
                                                 bootstrap_markers=full, published=published.get(claim, {}).get("stars"))}
        if paired:
            claims[claim]["paired_sensitivity"] = paired_comparison(evaluated[high]["records"], evaluated[low]["records"])
    for claim, (high, low) in PAIR_CLAIMS.items():
        claims[claim] = {"higher_group": high, "lower_group": low, "unit": "(speaker, target, case) pair",
                         "entities": {high: evaluated[high]["group"]["speakers"], low: evaluated[low]["group"]["speakers"]},
                         "tests": compare_groups(evaluated[high], evaluated[low], rng, samples,
                                                 bootstrap_markers=full, published=None)}
    groups = {name: percent(value["group"]) for name, value in evaluated.items()}
    return {"thresholds": thresholds, "groups": groups, "claims": claims}, evaluated


def votes_cross_check(selected: dict[str, dict], cases: dict[str, dict]) -> dict:
    """conversations.json votes_side against cases.jsonl votes_side for the selected sessions."""
    result = Counter()
    for meta in selected.values():
        case = cases.get(meta["case_id"])
        if case is None:
            result["case_missing_in_cases_jsonl"] += 1
            continue
        mine = {k: int(v) for k, v in (meta.get("votes_side") or {}).items()}
        theirs = {k: int(v) for k, v in (case.get("votes_side") or {}).items()}
        result["agree" if mine == theirs else "disagree"] += 1
        advocates = {k: int(v["side"]) for k, v in (meta.get("advocates") or {}).items()}
        theirs_adv = {v["id"]: int(v["side"]) for v in (case.get("advocates") or {}).values()}
        result["advocate_sides_agree" if advocates == theirs_adv else "advocate_sides_disagree"] += 1
    return dict(sorted(result.items()))


def write_entities(path: Path, evaluated: dict, counts: dict) -> int:
    """Per-entity sufficient statistics and coordination values; no utterance text."""
    rows = 0
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for name in GROUPS:
            speaker_type, target_type = GROUPS[name]
            for record in evaluated[name]["records"]:
                key = record["key"]
                raw = counts[name][key].to_dict()
                row = {"analysis": name, "speaker": key[0], "speaker_type": speaker_type, "target_type": target_type,
                       "target": key[1] if len(key) == 3 else None, "case_id": key[-1],
                       "exchanges": record["exchanges"], "counts": {"a": raw["a"], "b": raw["b"], "ab": raw["ab"]},
                       "coordination": record["by_marker"], **{a: record[a] for a in AGGREGATES}, "scale": "raw"}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                rows += 1
    return rows


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v, digits) for v in value) + "]"
    return f"{value:.{digits}f}"


def write_report(summary: dict, output: Path) -> None:
    lines = ["# Echoes of power, Supreme Court results: machine-written tables", "",
             f"Source: [{PAPER_URL}]({PAPER_URL}). Coordination values are percent (x100). "
             "The lexicon is an authored open list, not LIWC: compare directions and significance, not levels.", ""]
    data = summary["data"]
    lines += [f"Selected sessions: {summary['coverage']['selected_conversations']} "
              f"({summary['coverage']['selected_distinct_case_ids']} case ids); exchanges kept: {data['exchanges_kept_total']}; "
              f"Justices: {data['distinct_justices']}; lawyers: {data['distinct_lawyers']}.", ""]
    for claim, result in summary["primary"]["claims"].items():
        lines += [f"## {claim}", "", f"Higher (hypothesis): `{result['higher_group']}`; lower: `{result['lower_group']}`. "
                  f"Entities: {result['entities']}.", "",
                  "| Quantity | n (high, low) | mean high % | mean low % | diff % | Welch t | p | bootstrap 95% CI % | published |",
                  "|---|---|---:|---:|---:|---:|---:|---|---|"]
        for quantity, row in result["tests"].items():
            lines.append(f"| {quantity} | {row['n']} | {_fmt(row['mean_pct'][0])} | {_fmt(row['mean_pct'][1])} | "
                         f"{_fmt(row['difference_pct'])} | {_fmt(row['welch']['t'], 2)} | {_fmt(row['welch']['p'], 4)} | "
                         f"{_fmt(row.get('bootstrap_95_ci_pct'))} | {row.get('published_stars') if row.get('published_stars') is not None else ''} |")
        if "paired_sensitivity" in result:
            lines += ["", "Paired sensitivity (entities present in both groups): " +
                      "; ".join(f"{q}: n={v['n_pairs']}, diff={_fmt(v.get('mean_difference_pct'))}%, p={_fmt(v['p'], 4)}"
                                for q, v in result["paired_sensitivity"].items() if q in AGGREGATES)]
        lines.append("")
    lines += ["## Sensitivities (aggregates only)", "", "| Analysis | Claim | Agg 1 diff % (p) | Agg 2 diff % (p) | Agg 3 diff % (p) |", "|---|---|---|---|---|"]
    for name, sensitivity in summary["sensitivities"].items():
        for claim, result in sensitivity["claims"].items():
            cells = [f"{_fmt(result['tests'][a]['difference_pct'])} ({_fmt(result['tests'][a]['welch']['p'], 4)})" for a in AGGREGATES]
            lines.append(f"| {name} | {claim} | " + " | ".join(cells) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run(config: dict | Path, output: Path, cache: Path = Path(".cache/convokit/supreme")) -> dict:
    config_path = None
    if not isinstance(config, dict):
        config_path = Path(config)
        config = load_config(config_path)
    output, cache = Path(output), Path(cache)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be absent or empty; preserve previous evidence")
    samples = int(config["statistics"]["bootstrap_samples"])
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    paths = verify_inputs(cache, config)
    start = time.perf_counter()
    terms = [read_term(paths[f"supreme-{term}.zip"], term) for term in config["terms"]]
    selected, coverage = select_conversations(terms, config)
    if not selected:
        raise ValueError("No selected conversations: check case_ids against the term files")
    counts, data = collect_counts(terms, selected, config)
    cross_check = votes_cross_check(selected, read_cases(paths["cases.jsonl"])) if "cases.jsonl" in paths else None
    rng = np.random.default_rng(int(config["statistics"]["seed"]))
    primary, evaluated = analyse(counts, config["thresholds"]["primary"], rng, samples, config, full=True)
    sensitivities = {}
    for name, thresholds in config["thresholds"].get("sensitivities", {}).items():
        sensitivities[name], _ = analyse(counts, thresholds, rng, samples, config, full=False)
    if config.get("case_id_matching", "aliased") == "aliased" and config.get("case_id_aliases"):
        alias_only = {alias for docket, alias in config["case_id_aliases"].items() if docket not in config["case_ids"] or alias not in config["case_ids"]}
        exact_counts = {name: {key: value for key, value in entities.items() if key[-1][5:] not in alias_only}
                        for name, entities in counts.items()}
        sensitivities["exact_notebook_case_ids"], _ = analyse(exact_counts, config["thresholds"]["primary"], rng, samples, config, full=False)
        sensitivities["exact_notebook_case_ids"]["excluded_dockets"] = sorted(alias_only)
    verdicts = {}
    for claim, result in primary["claims"].items():
        verdicts[claim] = {a: {"direction_reproduced": result["tests"][a]["direction_reproduced"],
                               "significant_0_05": result["tests"][a]["significant_0_05"]} for a in AGGREGATES}
        verdicts[claim]["markers_in_hypothesised_direction"] = sum(bool(result["tests"][m]["direction_reproduced"]) for m in COORDINATION_MARKERS)
        verdicts[claim]["markers_significant_0_05"] = sum(bool(result["tests"][m]["significant_0_05"]) for m in COORDINATION_MARKERS)
    summary = {"schema_version": 1, "paper": PAPER_URL,
               "status": "independent reproduction on public ConvoKit data with a non-LIWC lexicon; directions and significance comparable, absolute levels not",
               "measure": {"implementation": "regesh.bonding.synchrony", "markers": list(COORDINATION_MARKERS),
                           "lexicon_sizes": {m: len(LEXICON[m]) for m in COORDINATION_MARKERS},
                           "lexicon": "authored open function-word list, not LIWC"},
               "p_value_engine": "scipy.stats.t" if scipy_stats is not None else "normal approximation",
               "coverage": coverage, "data": data, "votes_cross_check_against_cases_jsonl": cross_check,
               "published_data": config.get("published", {}).get("data"),
               "primary": primary, "verdicts": verdicts, "sensitivities": sensitivities}
    output.mkdir(parents=True, exist_ok=True)
    entity_rows = write_entities(output / "entities.jsonl.gz", evaluated, counts)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "runner.py").write_bytes(Path(__file__).read_bytes())
    write_report(summary, output)
    manifest = {"schema_version": 1, "search_date": config.get("search_date"), "paper": PAPER_URL,
                "inputs": {name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path),
                                  "used": config["inputs"][name].get("used", True)} for name, path in paths.items()},
                "case_set_source": config.get("case_set_source"),
                "config_path": str(config_path.resolve()) if config_path else None,
                "config_sha256": sha256(output / "config.json"), "source_code_sha256": sha256(Path(__file__)),
                "synchrony_source_sha256": sha256(Path(__file__).resolve().parents[1] / "bonding" / "synchrony.py"),
                "seed": config["statistics"]["seed"], "bootstrap_samples": samples,
                "elapsed_seconds": time.perf_counter() - start, "python": platform.python_version(),
                "numpy": np.__version__, "scipy": getattr(scipy_stats, "__name__", None) and __import__("scipy").__version__,
                "entity_records": entity_rows, "raw_text_written": False, "model_calls": 0,
                "unreproduced": ["LIWC dictionary (proprietary): replaced by an authored lexicon",
                                 "The authors' original transcript preprocessing [24] and exact exchange set",
                                 "Wikipedia results (Sections 5.1, 5.3 second half, 6)"],
                "artifacts": {name: sha256(output / name) for name in ("summary.json", "config.json", "runner.py", "report.md", "entities.jsonl.gz")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, default=Path(".cache/convokit/supreme"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.config, args.output, args.cache)
    print(json.dumps({"output": str(args.output), "coverage": summary["coverage"]["matched_exactly"],
                      "exchanges": summary["data"]["exchanges_kept_total"], "verdicts": summary["verdicts"]}, indent=1))


if __name__ == "__main__":
    main()
