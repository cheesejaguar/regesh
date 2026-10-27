"""Independent simulation reproduction of Hiolle, Lewis & Cañamero (2014).

*Arousal regulation and affective adaptation to human responsiveness by a robot
that explores and learns a novel environment*, Frontiers in Neurorobotics 8:17,
doi:10.3389/fnbot.2014.00017 (CC BY). Equations (1) and (3)-(7) and the
Table 1/2 parameters are imported from :mod:`regesh.bonding.attachment`; this
module never re-implements them. The original stimulation signal (Hopfield
recall error plus self-organizing-map weight change over camera contours,
Eq. 2) is unavailable and is replaced by a declared familiarity-based
substitute, so absolute pace and learning numbers are not comparable with the
paper. The Nao robot, the table and the automated caregiver are simulated.
No LLM, GPU or Regesh controller is involved; nothing here changes historical
results or trained weights.

Per-step order (declared; the paper's Algorithm 1 is a figure we could not
read): caregiver sensors -> Eq. 1 traces -> Eq. 4 comfort -> stimulation of the
perceived view -> Eq. 3 arousal -> behaviour selection -> request bookkeeping
-> Eq. 5-7 adaptation (adaptive profile only) -> motor action or learning.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from regesh.bonding import attachment
from regesh.bonding.attachment import (ADAPTIVE, COMFORT_FLOOR, PROFILES, arousal_step, behavior,
                                       comfort_step, perception_trace, profile_from_responsiveness,
                                       responsiveness_step)

PAPER = "https://doi.org/10.3389/fnbot.2014.00017"
BEHAVIOURS = ("explore", "learn", "request")
ENDPOINTS = ("requests", "responses", "requests_per_minute", "mean_arousal", "mean_stimulation",
             "frac_explore", "frac_learn", "frac_request", "total_learning", "positions_learned",
             "end_reached_s", "two_traversals_s", "comfort_duration_mean_s", "comfort_duration_max_s",
             "comfort_peak", "duration_s")
ADAPTIVE_ENDPOINTS = ("resp_150", "resp_end", "resp_min", "resp_max", "beta_h_150", "tau_h_150",
                      "beta_h_end", "tau_h_end", "mean_arousal_phase1", "mean_arousal_phase2",
                      "requests_phase1", "requests_phase2", "responses_phase1")

# Quoted published observations (prose of sections 3.1-3.3); figure values are approximate.
PUBLISHED = {
    "requests": "Needy requested comfort much more often than independent in both environments "
                "(Fig. 6A,B; section 3.2.2 'again ... the needy robot requested assistance more often').",
    "comfort_trace_independent": "Independent profile: Comf 'showing a trace lasting up to 10 s' (section 3.1).",
    "simple_pace": "Simple environment: 'both profiles go through the setup at a similar pace', end of table "
                   "around time step 160 (~50 s), back at the other end by step 250 (~83 s) (section 3.1).",
    "complex_pace_s": {"independent": 140.0, "needy": 170.0, "ratio_needy_over_independent": 170.0 / 140.0,
                       "quote": "the independent robot took 140 s to walk through the setup twice, the needy took 170 s"},
    "needy_learns_independent_explores": "'while the independent robot explored more, the needy robot learned more'; "
                                         "needy shows more frequent episodes of medium arousal (sections 3.1, 3.2.2).",
    "complex_vs_simple": "The independent profile 'did not solicit the attention of the caregiver more often than in "
                         "the simpler environment'; the needy 'produced more regulatory behaviors' (section 3.2.1).",
    "adaptive": "Responsiveness 'steadily increased (approximately from second 5-140)' under the responsive "
                "experimenter; after 150 s 'the responsiveness decreased' at the requests near 170 and 240 s; "
                "beta_h/tau_h moved toward needy then toward independent (section 3.3, Fig. 9).",
}


def environment_layout(config: dict, environment: str, rng: np.random.Generator) -> np.ndarray:
    """Objects visible from each of the lateral positions (declared substitute for the real table)."""
    low, high = config["simulation"]["environments"][environment]["n_objects_range"]
    return rng.integers(low, high + 1, size=config["simulation"]["positions"])


def stimulation(s_env: float, familiarity: float, n_objects: int, lam: float, noise: float,
                clip: tuple[float, float] = (0.0, 1.0)) -> float:
    """Declared substitute for Eq. 2: novelty decays with attending steps spent on a view."""
    value = s_env * math.exp(-familiarity / (lam * n_objects)) + noise
    return min(clip[1], max(clip[0], value))


def profile_parameters(profile: str) -> dict:
    """Table 1 fixed profiles, or Table 2 with Eq. 6-7 applied to Resp(0)."""
    if profile in PROFILES:
        return {**PROFILES[profile], "adaptive": False}
    if profile != "adaptive":
        raise ValueError("unknown profile")
    params = {**ADAPTIVE, "adaptive": True, "beta0": ADAPTIVE["beta_h"], "tau0": ADAPTIVE["tau_h"]}
    initial = profile_from_responsiveness(params["resp0"], params["beta0"], params["tau0"],
                                          params["r_beta"], params["r_tau"])
    params.update(beta_h=initial["beta_h"], tau_h=initial["tau_h"])
    return params


def caregiver_responds(cell: dict, config: dict, time_s: float) -> bool:
    policy = cell["caregiver"]
    if policy == "responsive":
        return True
    if policy == "nonresponsive":
        return False
    if policy == "schedule":
        return time_s < config["simulation"]["adaptive_experiment"]["responsive_until_s"]
    raise ValueError("unknown caregiver policy")


def cell_setting(cell: dict, config: dict, key: str):
    """Cell overrides take precedence over the shared simulation block."""
    return cell.get("overrides", {}).get(key, config["simulation"][key])


def simulate_run(config: dict, cell: dict, seed: int, keep_series: bool = False) -> tuple[dict, dict | None]:
    """One robot run. Same seed + environment => same object layout and noise across cells."""
    sim = config["simulation"]
    sps = sim["steps_per_second"]
    dt = 1.0 / sps
    positions = sim["positions"]
    env_name = cell["environment"]
    env = sim["environments"][env_name]
    env_index = list(sim["environments"]).index(env_name)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), env_index]))
    n_objects = environment_layout(config, env_name, rng)
    noise_steps = int(round(sim["noise_horizon_s"] * sps))
    noise = rng.normal(0.0, sim["stimulation_noise_sd"], noise_steps)
    if cell["experiment"] == "adaptive":
        horizon = sim["adaptive_experiment"]["duration_s"]
        stop_after_two = False
    else:
        horizon = sim["profile_experiments"]["max_duration_s"]
        stop_after_two = sim["profile_experiments"]["stop_after_two_traversals"]
    max_steps = int(round(horizon * sps))
    if max_steps > noise_steps:
        raise ValueError("noise horizon shorter than the run horizon")
    params = profile_parameters(cell["profile"])
    move_steps = int(cell_setting(cell, config, "move_steps"))
    latency = int(cell_setting(cell, config, "latency_steps"))
    intervention = int(cell_setting(cell, config, "intervention_steps"))
    comfort_condition = cell_setting(cell, config, "comfort_condition")
    semantics = cell_setting(cell, config, "regulatory_semantics")
    transit_view = cell_setting(cell, config, "transit_view")
    learning_states = set(sim["learning_states"])
    alpha_sensor, lam, floor = sim["alpha_sensor"], sim["lambda"], sim["arousal_floor"]
    clip = tuple(sim["stimulation_clip"])
    phase_split_s = sim["adaptive_experiment"]["responsive_until_s"]
    phase_split_step = int(round(phase_split_s * sps))
    if comfort_condition not in {"sensor", "trace"} or semantics not in {"literal", "episode"}:
        raise ValueError("undeclared comfort condition or regulatory semantics")

    position, direction, in_transit, destination, progress = 1, 1, False, None, 0
    arousal, comfort, face_trace, contact_trace = sim["initial_arousal"], 0.0, 0.0, 0.0
    beta_h, tau_h, resp = params["beta_h"], params["tau_h"], params.get("resp0")
    familiarity = np.zeros(positions + 1)
    pending_start = None
    previous = None
    requests = responses = learning = 0
    band = {name: 0 for name in BEHAVIOURS}
    arousal_sum = stim_sum = 0.0
    phase = {"arousal": [0.0, 0.0], "steps": [0, 0], "requests": [0, 0], "responses": [0, 0]}
    end_step = two_step = None
    comfort_since = None
    comfort_spans: list[int] = []
    comfort_peak = 0.0
    snapshot_150 = None
    resp_min = resp_max = resp
    series = {k: [] for k in ("time_s", "position", "in_transit", "stimulation", "arousal", "comfort",
                              "face_trace", "contact_trace", "behaviour", "request_onset",
                              "intervention_active", "responsiveness", "beta_h", "tau_h")} if keep_series else None
    steps = max_steps
    for t in range(max_steps):
        time_s = t * dt
        active = pending_start is not None and pending_start <= t < pending_start + intervention
        sensor = 1 if active else 0
        face_trace = perception_trace(face_trace, sensor, alpha_sensor)
        contact_trace = perception_trace(contact_trace, sensor, alpha_sensor)
        if active and t == pending_start:
            responses += 1
            phase["responses"][int(t >= phase_split_step)] += 1
        if comfort_condition == "sensor":
            contact_in, face_in = (contact_trace, face_trace) if active else (0.0, 0.0)
        else:
            contact_in, face_in = contact_trace, face_trace
        comfort = comfort_step(comfort, contact_in, face_in, tau_h, beta_h)
        comfort_peak = max(comfort_peak, comfort)
        if comfort > COMFORT_FLOOR and comfort_since is None:
            comfort_since = t
        elif comfort <= COMFORT_FLOOR and comfort_since is not None:
            comfort_spans.append(t - comfort_since)
            comfort_since = None
        view = destination if (in_transit and transit_view == "destination") else position
        stim = stimulation(env["s_env"], familiarity[view], int(n_objects[view - 1]), lam, noise[t], clip)
        arousal = arousal_step(arousal, stim, comfort, params["tau_ar"], params["alpha_ar"], floor)
        beh = behavior(arousal, params["high"], params["low"])
        onset = beh == "request" and previous != "request"
        if onset:
            requests += 1
            phase["requests"][int(t >= phase_split_step)] += 1
            if pending_start is None and caregiver_responds(cell, config, time_s):
                pending_start = t + latency
        if params["adaptive"]:
            regulatory_active = beh == "request" or (semantics == "episode" and active)
            if regulatory_active:
                resp = responsiveness_step(resp, comfort, params["alpha_resp"])
                updated = profile_from_responsiveness(resp, params["beta0"], params["tau0"],
                                                      params["r_beta"], params["r_tau"])
                beta_h, tau_h = updated["beta_h"], updated["tau_h"]
                resp_min, resp_max = min(resp_min, resp), max(resp_max, resp)
        if in_transit:
            progress += 1
            if progress >= move_steps:
                position, in_transit, destination = destination, False, None
        elif beh == "explore":
            if not 1 <= position + direction <= positions:
                direction = -direction
            destination, progress, in_transit = position + direction, 1, True
            if progress >= move_steps:
                position, in_transit, destination = destination, False, None
        elif beh in learning_states:
            familiarity[position] += 1
            learning += 1
        if pending_start is not None and t >= pending_start + intervention - 1:
            pending_start = None
        band[beh] += 1
        arousal_sum += arousal
        stim_sum += stim
        p = int(t >= phase_split_step)
        phase["arousal"][p] += arousal
        phase["steps"][p] += 1
        if position == positions and end_step is None:
            end_step = t
        if end_step is not None and t > end_step and position == 1 and two_step is None:
            two_step = t
        if t + 1 == phase_split_step:
            snapshot_150 = (resp, beta_h, tau_h)
        if keep_series:
            series["time_s"].append(time_s)
            series["position"].append(position)
            series["in_transit"].append(in_transit)
            series["stimulation"].append(stim)
            series["arousal"].append(arousal)
            series["comfort"].append(comfort)
            series["face_trace"].append(face_trace)
            series["contact_trace"].append(contact_trace)
            series["behaviour"].append(BEHAVIOURS.index(beh))
            series["request_onset"].append(onset)
            series["intervention_active"].append(active)
            series["responsiveness"].append(np.nan if resp is None else resp)
            series["beta_h"].append(beta_h)
            series["tau_h"].append(tau_h)
        previous = beh
        if stop_after_two and two_step is not None:
            steps = t + 1
            break
    duration_s = steps * dt
    comfort_peak, arousal_sum, stim_sum = float(comfort_peak), float(arousal_sum), float(stim_sum)
    record = {
        "cell": cell["name"], "experiment": cell["experiment"], "environment": env_name,
        "profile": cell["profile"], "caregiver": cell["caregiver"], "seed": int(seed),
        "move_steps": move_steps, "comfort_condition": comfort_condition, "regulatory_semantics": semantics,
        "intervention_steps": intervention, "n_objects": [int(x) for x in n_objects],
        "final_familiarity": [float(x) for x in familiarity[1:]],
        "steps": steps, "duration_s": duration_s, "censored": two_step is None,
        "requests": requests, "responses": responses, "requests_per_minute": requests / (duration_s / 60.0),
        "mean_arousal": arousal_sum / steps, "mean_stimulation": stim_sum / steps,
        "frac_explore": band["explore"] / steps, "frac_learn": band["learn"] / steps,
        "frac_request": band["request"] / steps, "total_learning": learning,
        "positions_learned": int((familiarity[1:] > 0).sum()),
        "end_reached_s": None if end_step is None else (end_step + 1) * dt,
        "two_traversals_s": None if two_step is None else (two_step + 1) * dt,
        "comfort_duration_mean_s": None if not comfort_spans else float(np.mean(comfort_spans)) * dt,
        "comfort_duration_max_s": None if not comfort_spans else max(comfort_spans) * dt,
        "comfort_spans_completed": len(comfort_spans), "comfort_span_open_at_end": comfort_since is not None,
        "comfort_peak": comfort_peak,
    }
    if params["adaptive"]:
        resp_150, beta_150, tau_150 = snapshot_150 if snapshot_150 else (None, None, None)
        record.update({
            "resp_150": resp_150, "resp_end": resp, "resp_min": resp_min, "resp_max": resp_max,
            "beta_h_150": beta_150, "tau_h_150": tau_150, "beta_h_end": beta_h, "tau_h_end": tau_h,
            "mean_arousal_phase1": phase["arousal"][0] / phase["steps"][0] if phase["steps"][0] else None,
            "mean_arousal_phase2": phase["arousal"][1] / phase["steps"][1] if phase["steps"][1] else None,
            "requests_phase1": phase["requests"][0], "requests_phase2": phase["requests"][1],
            "responses_phase1": phase["responses"][0]})
    record = {k: (float(v) if isinstance(v, float) else v) for k, v in record.items()}
    if keep_series:
        series = {k: np.asarray(v) for k, v in series.items()}
    return record, series


def interval(values: np.ndarray, rng: np.random.Generator, samples: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    boot = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def contrast(a: list, b: list, rng: np.random.Generator, samples: int, paired: bool) -> dict:
    """Mean of a minus mean of b; paired resamples differences, independent resamples both groups."""
    if paired:
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if not pairs:
            return {"n": 0, "difference": None, "ci95": None, "paired": True}
        diff = np.array([x - y for x, y in pairs], dtype=float)
        return {"n": len(pairs), "difference": float(diff.mean()), "ci95": interval(diff, rng, samples), "paired": True}
    xa = np.array([x for x in a if x is not None], dtype=float)
    xb = np.array([x for x in b if x is not None], dtype=float)
    if not len(xa) or not len(xb):
        return {"n_a": len(xa), "n_b": len(xb), "difference": None, "ci95": None, "paired": False}
    boot = (xa[rng.integers(0, len(xa), (samples, len(xa)))].mean(axis=1)
            - xb[rng.integers(0, len(xb), (samples, len(xb)))].mean(axis=1))
    return {"n_a": len(xa), "n_b": len(xb), "difference": float(xa.mean() - xb.mean()),
            "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])], "paired": False}


def describe(values: list) -> dict:
    present = np.array([v for v in values if v is not None], dtype=float)
    out = {"n": len(values), "n_observed": int(len(present)), "censored": int(len(values) - len(present))}
    if len(present):
        out.update(mean=float(present.mean()), sd=float(present.std(ddof=1)) if len(present) > 1 else 0.0,
                   min=float(present.min()), max=float(present.max()), median=float(np.median(present)))
    return out


def seeds_of(config: dict, cell: dict) -> dict[str, list[int]]:
    return {name: list(config["seeds"][name]) for name in cell["seed_sets"]}


def cell_records(records: list[dict], name: str, seed_set: list[int]) -> list[dict]:
    by_seed = {r["seed"]: r for r in records if r["cell"] == name}
    return [by_seed[s] for s in seed_set if s in by_seed]


def direction_status(result: dict, expected_sign: int) -> str:
    ci = result.get("ci95")
    if ci is None:
        return "not evaluable"
    low, high = ci
    if expected_sign > 0 and low > 0:
        return "direction reproduced (95% interval excludes zero)"
    if expected_sign < 0 and high < 0:
        return "direction reproduced (95% interval excludes zero)"
    if (expected_sign > 0 and high < 0) or (expected_sign < 0 and low > 0):
        return "opposite direction (95% interval excludes zero)"
    return "direction not established (95% interval includes zero)"


def analyze(records: list[dict], config: dict) -> dict:
    rng = np.random.default_rng(config["seeds"]["bootstrap_resampling"])
    samples = config["bootstrap_samples"]
    boot = config["seeds"]["bootstrap"]
    cells = {c["name"]: c for c in config["cells"]}
    summary = {"cells": [], "contrasts": [], "adaptive": [], "published_comparisons": []}
    for cell in config["cells"]:
        for set_name, seeds in seeds_of(config, cell).items():
            rows = cell_records(records, cell["name"], seeds)
            if not rows:
                continue
            entry = {"cell": cell["name"], "role": cell["role"], "seed_set": set_name, "runs": len(rows),
                     "experiment": cell["experiment"], "environment": cell["environment"],
                     "profile": cell["profile"], "caregiver": cell["caregiver"],
                     "overrides": cell.get("overrides", {}),
                     "endpoints": {k: describe([r[k] for r in rows]) for k in ENDPOINTS}}
            if cell["profile"] == "adaptive":
                entry["adaptive_endpoints"] = {k: describe([r[k] for r in rows]) for k in ADAPTIVE_ENDPOINTS}
                entry["fraction_resp_150_above_initial"] = float(np.mean([r["resp_150"] > ADAPTIVE["resp0"] for r in rows]))
                entry["fraction_resp_end_below_150"] = float(np.mean([r["resp_end"] < r["resp_150"] for r in rows]))
            summary["cells"].append(entry)

    def rows_for(name):
        return cell_records(records, name, boot)

    paired_keys = ["requests", "requests_per_minute", "mean_arousal", "frac_explore", "frac_learn", "frac_request",
                   "total_learning", "end_reached_s", "two_traversals_s", "comfort_duration_mean_s",
                   "comfort_duration_max_s", "duration_s"]
    for spec in config["contrasts"]:
        a, b = rows_for(spec["a"]), rows_for(spec["b"])
        if not a or not b:
            continue
        out = {"name": spec["name"], "a": spec["a"], "b": spec["b"], "paired": spec["paired"],
               "runs_a": len(a), "runs_b": len(b), "endpoints": {}}
        for key in paired_keys:
            out["endpoints"][key] = contrast([r[key] for r in a], [r[key] for r in b], rng, samples, spec["paired"])
            out["endpoints"][key]["censored_a"] = sum(r[key] is None for r in a)
            out["endpoints"][key]["censored_b"] = sum(r[key] is None for r in b)
        ta = [r["two_traversals_s"] for r in a if r["two_traversals_s"] is not None]
        tb = [r["two_traversals_s"] for r in b if r["two_traversals_s"] is not None]
        out["two_traversals_ratio_of_means_a_over_b"] = (float(np.mean(ta) / np.mean(tb)) if ta and tb else None)
        summary["contrasts"].append(out)

    for cell in config["cells"]:
        if cell["profile"] != "adaptive":
            continue
        rows = rows_for(cell["name"])
        if not rows:
            continue
        r150 = np.array([r["resp_150"] for r in rows], dtype=float)
        rend = np.array([r["resp_end"] for r in rows], dtype=float)
        summary["adaptive"].append({
            "cell": cell["name"], "role": cell["role"], "runs": len(rows), "overrides": cell.get("overrides", {}),
            "resp_150_minus_initial": {"difference": float((r150 - ADAPTIVE["resp0"]).mean()),
                                       "ci95": interval(r150 - ADAPTIVE["resp0"], rng, samples)},
            "resp_end_minus_resp_150": {"difference": float((rend - r150).mean()), "ci95": interval(rend - r150, rng, samples)},
            "mean_resp_150": float(r150.mean()), "mean_resp_end": float(rend.mean()),
            "mean_beta_h_150": float(np.mean([r["beta_h_150"] for r in rows])),
            "mean_tau_h_150": float(np.mean([r["tau_h_150"] for r in rows])),
            "mean_beta_h_end": float(np.mean([r["beta_h_end"] for r in rows])),
            "mean_tau_h_end": float(np.mean([r["tau_h_end"] for r in rows])),
            "mean_arousal_phase1": float(np.mean([r["mean_arousal_phase1"] for r in rows])),
            "mean_arousal_phase2": float(np.mean([r["mean_arousal_phase2"] for r in rows])),
            "mean_requests_phase1": float(np.mean([r["requests_phase1"] for r in rows])),
            "mean_requests_phase2": float(np.mean([r["requests_phase2"] for r in rows])),
            "fraction_resp_150_above_initial": float(np.mean(r150 > ADAPTIVE["resp0"])),
            "fraction_resp_end_below_150": float(np.mean(rend < r150))})

    contrasts = {c["name"]: c for c in summary["contrasts"]}
    adaptive = {a["cell"]: a for a in summary["adaptive"]}
    cells_by = {c["cell"]: c for c in summary["cells"] if c["seed_set"] == "bootstrap"}
    comparisons = []
    for env in ("simple", "complex"):
        c = contrasts.get(f"{env}_needy_minus_independent_responsive")
        if c:
            comparisons.append({"target": "requests", "environment": env, "published": PUBLISHED["requests"],
                                "measured": c["endpoints"]["requests"],
                                "status": direction_status(c["endpoints"]["requests"], +1)})
            comparisons.append({"target": "needy_more_medium_arousal", "environment": env,
                                "published": PUBLISHED["needy_learns_independent_explores"],
                                "measured": c["endpoints"]["frac_learn"],
                                "status": direction_status(c["endpoints"]["frac_learn"], +1)})
            comparisons.append({"target": "independent_explores_more", "environment": env,
                                "published": PUBLISHED["needy_learns_independent_explores"],
                                "measured": c["endpoints"]["frac_explore"],
                                "status": direction_status(c["endpoints"]["frac_explore"], -1)})
            comparisons.append({"target": "needy_learns_more", "environment": env,
                                "published": PUBLISHED["needy_learns_independent_explores"],
                                "measured": c["endpoints"]["total_learning"],
                                "status": direction_status(c["endpoints"]["total_learning"], +1)})
    ind = cells_by.get("simple_independent_responsive")
    if ind:
        d = ind["endpoints"]["comfort_duration_max_s"]
        comparisons.append({"target": "comfort_trace_independent", "environment": "simple",
                            "published": PUBLISHED["comfort_trace_independent"], "measured": d,
                            "status": ("order of magnitude consistent (mean per-run maximum within 5-15 s)"
                                       if d.get("mean") is not None and 5 <= d["mean"] <= 15 else "not consistent"),
                            "note": "Duration follows from beta_h=.95 and the 3-step intervention at dt=1/3 s; "
                                    "it is an arithmetic consequence of Eq. 4, not an independent observation."})
    simple = contrasts.get("simple_needy_minus_independent_responsive")
    complex_ = contrasts.get("complex_needy_minus_independent_responsive")
    if simple and complex_:
        rs, rc = simple["two_traversals_ratio_of_means_a_over_b"], complex_["two_traversals_ratio_of_means_a_over_b"]
        comparisons.append({"target": "complex_pace", "environment": "complex",
                            "published": PUBLISHED["complex_pace_s"],
                            "measured": {"ratio_needy_over_independent": rc,
                                         "paired_difference_s": complex_["endpoints"]["two_traversals_s"]},
                            "status": direction_status(complex_["endpoints"]["two_traversals_s"], +1)
                                      + "; absolute seconds not comparable (substituted stimulation)"})
        comparisons.append({"target": "simple_pace_similar", "environment": "simple",
                            "published": PUBLISHED["simple_pace"],
                            "measured": {"ratio_needy_over_independent": rs,
                                         "paired_difference_s": simple["endpoints"]["two_traversals_s"],
                                         "complex_ratio_for_reference": rc},
                            "status": ("profile gap smaller in simple than complex environment"
                                       if rs is not None and rc is not None and abs(rs - 1) < abs(rc - 1)
                                       else "profile gap not smaller in simple environment")
                                      + "; absolute 50 s / 83 s not comparable (substituted stimulation)"})
    for profile, sign in (("needy", +1), ("independent", 0)):
        c = contrasts.get(f"{profile}_complex_minus_simple_responsive")
        if c:
            comparisons.append({"target": "complex_vs_simple_requests", "profile": profile,
                                "published": PUBLISHED["complex_vs_simple"], "measured": c["endpoints"]["requests"],
                                "status": (direction_status(c["endpoints"]["requests"], +1) if sign else
                                           ("no increase established (interval includes zero or negative)"
                                            if c["endpoints"]["requests"]["ci95"] and c["endpoints"]["requests"]["ci95"][0] <= 0
                                            else "increase found; paper reports none"))})
    for name, a in adaptive.items():
        comparisons.append({"target": "adaptive_responsiveness_rises_then_falls", "cell": name, "role": a["role"],
                            "published": PUBLISHED["adaptive"],
                            "measured": {"resp_150_minus_initial": a["resp_150_minus_initial"],
                                         "resp_end_minus_resp_150": a["resp_end_minus_resp_150"]},
                            "status_phase1_increase": direction_status(a["resp_150_minus_initial"], +1),
                            "status_phase2_decrease": direction_status(a["resp_end_minus_resp_150"], -1)})
    summary["published_comparisons"] = comparisons
    summary["interpretation"] = (
        "Independent simulation of the published architecture with a declared familiarity-based stimulation "
        "substitute, simulated Nao movement and an automated caregiver. Directions are compared with the paper's "
        "prose; absolute seconds, request counts and learning totals are not comparable. Intervals are nominal "
        "95% percentile bootstraps over seeds (paired by seed where the same environment layout and noise were "
        "used), without multiplicity correction. Nothing here establishes affect, biological fidelity or an "
        "advantage of any Regesh controller.")
    return summary


def pilot_calibration(config: dict) -> dict:
    """Re-run the declared pilot seed; band occupancy was the only calibration criterion."""
    out = {"seed": config["seeds"]["pilot"], "criterion": config["pilot"]["criterion"], "runs": []}
    for env in config["simulation"]["environments"]:
        for profile in ("independent", "needy"):
            cell = {"name": f"pilot_{env}_{profile}", "experiment": env, "environment": env,
                    "profile": profile, "caregiver": "responsive"}
            record, _ = simulate_run(config, cell, config["seeds"]["pilot"])
            out["runs"].append({"environment": env, "profile": profile,
                                **{k: record[k] for k in ("frac_explore", "frac_learn", "frac_request", "requests",
                                                          "mean_stimulation", "two_traversals_s", "censored")},
                                "all_three_bands_visited": all(record[f"frac_{b}"] > 0 for b in BEHAVIOURS)})
    out["criterion_met_for_independent"] = all(r["all_three_bands_visited"] for r in out["runs"] if r["profile"] == "independent")
    return out


def run(config_path: str | Path, output: str | Path) -> dict:
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; preserve prior measurements")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    source = Path(__file__).read_bytes()
    (output / "source.py").write_bytes(source)
    equations = Path(attachment.__file__).read_bytes()
    (output / "equations_source.py").write_bytes(equations)
    start = time.perf_counter()
    pilot = pilot_calibration(config)
    records: list[dict] = []
    total_steps = 0
    for cell in config["cells"]:
        representative = None
        for set_name, seeds in seeds_of(config, cell).items():
            for seed in seeds:
                keep = representative is None
                record, series = simulate_run(config, cell, seed, keep_series=keep)
                records.append(record)
                total_steps += record["steps"]
                if keep:
                    representative = seed
                    np.savez_compressed(output / f"series_{cell['name']}.npz", seed=np.int64(seed), **series)
        print(f"completed {cell['name']}: {sum(len(s) for s in seeds_of(config, cell).values())} runs", flush=True)
    summary = analyze(records, config)
    summary["pilot_calibration"] = pilot
    summary["runs"] = len(records)
    summary["simulated_steps"] = total_steps
    summary["simulated_robot_seconds"] = total_steps / config["simulation"]["steps_per_second"]
    summary["runtime_seconds"] = time.perf_counter() - start
    with gzip.open(output / "runs.jsonl.gz", "wt") as f:
        for row in records:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "paper_url": PAPER, "license": "CC BY (Frontiers)",
        "author_code_executed": False, "author_code_copied": False,
        "fidelity": "independent simulation of the published architecture with declared stimulation, movement "
                    "and caregiver substitutes; equations imported from regesh.bonding.attachment",
        "python": platform.python_version(), "numpy": np.__version__,
        "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "equations_source_sha256": hashlib.sha256(equations).hexdigest(),
        "command": f"python -m regesh.reproductions.attachment_robot --config {config_path} --output {output}",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reproductions/attachment_robot.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
