"""Computational reproduction of the Gottman dyadic affect model.

Gottman, Swanson & Murray (1999), J. Family Psychology 13(1):3-19, and the
repair/damping extension of Gottman et al. (2003), J. Homosexuality 45(1):65-91.
The coupled difference equations, influence forms, steady-state solvers and the
published estimation algorithm are imported from regesh.bonding.dyad and are not
re-implemented here. Four prespecified experiments use only published group-mean
parameters and declared choices; no couple data are involved. Nothing here
estimates feelings, and no Regesh controller or weight is touched.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from regesh.bonding import dyad
from regesh.bonding.dyad import PUBLISHED_1999, SPAFF_RANGE

GROUPS = ("stable", "divorced", "happy_stable", "unhappy_stable")
QUADRANTS = ("++", "+-", "-+", "--")
ESTIMATED = ("inertia", "uninfluenced", "neg_threshold", "pos_threshold", "neg_value", "pos_value")


# ----------------------------------------------------------------------------- couples
def autoregressive(group: str) -> dict:
    """a, r1, b, r2 from the published inertia and uninfluenced steady state: a = u (1 - r)."""
    wife, husband = PUBLISHED_1999[group]["wife"], PUBLISHED_1999[group]["husband"]
    return {"a": wife["uninfluenced"] * (1 - wife["inertia"]), "r1": wife["inertia"],
            "b": husband["uninfluenced"] * (1 - husband["inertia"]), "r2": husband["inertia"]}


def ojive_spec(neg_threshold, pos_threshold, plateau_neg, plateau_pos, form="ojive", width=None) -> dict:
    """Plateau magnitudes are positive numbers; the negative plateau is -|plateau_neg|."""
    spec = {"form": form, "neg_threshold": float(neg_threshold), "neg_value": -abs(float(plateau_neg)),
            "pos_threshold": float(pos_threshold), "pos_value": abs(float(plateau_pos))}
    if form == "sigmoid":
        spec["width"] = float(width)
    return spec


def published_couple(group: str, plateau_neg, plateau_pos, form="ojive", width=None) -> dict:
    """Both spouses from PUBLISHED_1999.

    I_hw, the husband's influence on the wife, is a function of H and carries the
    husband's published thresholds; I_wh carries the wife's (declared mapping).
    """
    wife, husband = PUBLISHED_1999[group]["wife"], PUBLISHED_1999[group]["husband"]
    params = autoregressive(group)
    params["I_hw"] = ojive_spec(husband["neg_threshold"], husband["pos_threshold"], plateau_neg, plateau_pos, form, width)
    params["I_wh"] = ojive_spec(wife["neg_threshold"], wife["pos_threshold"], plateau_neg, plateau_pos, form, width)
    return params


def grid_couple(thresholds: dict, r_w, r_h, u_w, u_h, plateau_neg, plateau_pos, form="ojive", width=None) -> dict:
    return {"a": u_w * (1 - r_w), "r1": r_w, "b": u_h * (1 - r_h), "r2": r_h,
            "I_hw": ojive_spec(thresholds["I_hw"]["neg"], thresholds["I_hw"]["pos"], plateau_neg, plateau_pos, form, width),
            "I_wh": ojive_spec(thresholds["I_wh"]["neg"], thresholds["I_wh"]["pos"], plateau_neg, plateau_pos, form, width)}


def repair_damping_couple(group: str, slopes, repair=None, damping=None) -> dict:
    """2003 bilinear form with optional constant repair/damping triggers, identical for both spouses."""
    params = autoregressive(group)
    for key in ("I_hw", "I_wh"):
        spec = {"form": "bilinear", "slope_neg": float(slopes[0]), "slope_pos": float(slopes[1])}
        if repair is not None:
            spec["repair"] = {"effectiveness": float(repair[0]), "threshold": float(repair[1])}
        if damping is not None:
            spec["damping"] = {"effectiveness": float(damping[0]), "threshold": float(damping[1])}
        params[key] = spec
    return params


# ----------------------------------------------------------------------------- E1 helpers
def piece_label(x: float, spec: dict) -> str:
    if x <= spec["neg_threshold"]:
        return "N"
    if x >= spec["pos_threshold"]:
        return "P"
    return "Z"


def step_nullclines(params: dict) -> dict:
    """Strict fixed points of the step-ojive map plus jump crossings of the completed null clines.

    The wife's null cline W = (a + I_hw(H))/(1 - r1) has three levels and jumps at the
    thresholds of I_hw; the husband's H = (b + I_wh(W))/(1 - r2) likewise. Every strict
    fixed point is a level-level crossing. Drawing each jump as a connecting segment,
    as the paper's sigmoid figure does, adds a crossing wherever a wife jump (H = K,
    W within its span) meets a husband jump (W = K', H within its span). Those are the
    paper's graphically unstable steady states; they are not fixed points of the step
    map and `dyad.steady_states` excludes them.
    """
    strict = dyad.steady_states(params)
    ihw, iwh = params["I_hw"], params["I_wh"]
    level_w = {k: (params["a"] + v) / (1 - params["r1"]) for k, v in (("N", ihw["neg_value"]), ("Z", 0.), ("P", ihw["pos_value"]))}
    level_h = {k: (params["b"] + v) / (1 - params["r2"]) for k, v in (("N", iwh["neg_value"]), ("Z", 0.), ("P", iwh["pos_value"]))}
    jumps_w = [(ihw["neg_threshold"], sorted((level_w["N"], level_w["Z"]))), (ihw["pos_threshold"], sorted((level_w["Z"], level_w["P"])))]
    jumps_h = [(iwh["neg_threshold"], sorted((level_h["N"], level_h["Z"]))), (iwh["pos_threshold"], sorted((level_h["Z"], level_h["P"])))]
    crossings = sum(1 for kh, (wlo, whi) in jumps_w for kw, (hlo, hhi) in jumps_h if wlo <= kw <= whi and hlo <= kh <= hhi)
    degenerate = (any(abs(level - k) < 1e-9 for level in level_w.values() for k in (iwh["neg_threshold"], iwh["pos_threshold"]))
                  or any(abs(level - k) < 1e-9 for level in level_h.values() for k in (ihw["neg_threshold"], ihw["pos_threshold"])))
    return {"fixed_points": strict, "n_fixed": len(strict), "n_stable": int(sum(s["stable"] for s in strict)),
            "types": ["".join((piece_label(s["w"], iwh), piece_label(s["h"], ihw))) for s in strict],
            "n_jump_crossings": int(crossings), "n_completed": len(strict) + int(crossings), "degenerate": bool(degenerate)}


def predicted_step_types(params: dict) -> list[str]:
    """Closed-form existence of the NN, ZZ and PP fixed points of the step map (a check on the solver).

    With u = a/(1-r) and plateau v, the candidate on the (w-piece, h-piece) pair (X, Y)
    is w = u_w + I_hw(Y)/(1-r1), h = u_h + I_wh(X)/(1-r2); it exists when w lies in piece
    X of I_wh and h in piece Y of I_hw.
    """
    ihw, iwh = params["I_hw"], params["I_wh"]
    value = {"N": lambda s: s["neg_value"], "Z": lambda s: 0., "P": lambda s: s["pos_value"]}
    out = []
    for x, y in itertools.product("NZP", "NZP"):
        w = (params["a"] + value[y](ihw)) / (1 - params["r1"])
        h = (params["b"] + value[x](iwh)) / (1 - params["r2"])
        if piece_label(w, iwh) == x and piece_label(h, ihw) == y:
            out.append(x + y)
    return out


def numeric_fixed_points(params: dict, seed_grids: list[dict]) -> list[dict]:
    """Union of `dyad.fixed_points_numeric` runs from the declared seed grids."""
    found: list[dict] = []
    for grid in seed_grids:
        for s in dyad.fixed_points_numeric(params, box=tuple(grid["box"]), seeds=int(grid["seeds"])):
            if not any(abs(s["w"] - f["w"]) < 1e-5 and abs(s["h"] - f["h"]) < 1e-5 for f in found):
                found.append(s)
    found.sort(key=lambda s: (s["w"], s["h"]))
    return found


def smooth_combos(cfg: dict) -> list[tuple]:
    combos = [(cfg["smooth_primary_width"], r, u_w, u_h, m_n, m_p) for r, u_w, u_h, m_n, m_p in itertools.product(
        cfg["inertia_grid"], cfg["smooth_uninfluenced_grid"], cfg["smooth_uninfluenced_grid"],
        cfg["plateau_magnitudes"], cfg["plateau_magnitudes"])]
    combos += [(width, cfg["smooth_sensitivity_inertia"], u_w, u_h, m_n, m_p) for width, u_w, u_h, m_n, m_p in itertools.product(
        cfg["smooth_sensitivity_widths"], cfg["smooth_sensitivity_uninfluenced_grid"], cfg["smooth_sensitivity_uninfluenced_grid"],
        cfg["plateau_magnitudes"], cfg["plateau_magnitudes"])]
    return combos


def e1_step_records(cfg: dict) -> list[dict]:
    records = []
    for r_w, r_h, u_w, u_h, m_n, m_p in itertools.product(cfg["inertia_grid"], cfg["inertia_grid"], cfg["uninfluenced_grid"],
                                                          cfg["uninfluenced_grid"], cfg["plateau_magnitudes"], cfg["plateau_magnitudes"]):
        params = grid_couple(cfg["thresholds"], r_w, r_h, u_w, u_h, m_n, m_p)
        res = step_nullclines(params)
        records.append({"form": "step", "r_w": r_w, "r_h": r_h, "u_w": u_w, "u_h": u_h, "plateau_neg": m_n, "plateau_pos": m_p,
                        "neg_ge_pos": m_n >= m_p, "n_fixed": res["n_fixed"], "n_stable": res["n_stable"], "types": res["types"],
                        "predicted_types": predicted_step_types(params), "n_jump_crossings": res["n_jump_crossings"],
                        "n_completed": res["n_completed"], "degenerate": res["degenerate"],
                        "fixed_points": [[s["w"], s["h"], s["stable"]] for s in res["fixed_points"]]})
    return records


def _e1_smooth_task(args) -> list[dict]:
    cfg, combos = args
    out = []
    for width, r, u_w, u_h, m_n, m_p in combos:
        fps = numeric_fixed_points(grid_couple(cfg["thresholds"], r, r, u_w, u_h, m_n, m_p, "sigmoid", width),
                                   cfg["numeric_fixed_points"]["seed_grids"])
        step = step_nullclines(grid_couple(cfg["thresholds"], r, r, u_w, u_h, m_n, m_p))
        out.append({"form": "sigmoid", "width": width, "r": r, "u_w": u_w, "u_h": u_h, "plateau_neg": m_n, "plateau_pos": m_p,
                    "neg_ge_pos": m_n >= m_p, "n_fixed": len(fps), "n_stable": int(sum(s["stable"] for s in fps)),
                    "n_unstable": int(sum(not s["stable"] for s in fps)), "even_count": len(fps) % 2 == 0,
                    "step_n_fixed": step["n_fixed"], "step_n_completed": step["n_completed"],
                    "agrees_with_step_completed": len(fps) == step["n_completed"],
                    "fixed_points": [[s["w"], s["h"], s["stable"], s["spectral_radius"]] for s in fps]})
    return out


def _e1_reference_task(args) -> list[dict]:
    cfg, combos = args
    out = []
    for width, r, u_w, u_h, m_n, m_p in combos:
        params = grid_couple(cfg["thresholds"], r, r, u_w, u_h, m_n, m_p, "sigmoid", width)
        reference = dyad.fixed_points_numeric(params, seeds=int(cfg["numeric_fixed_points"]["reference_seeds"]))
        primary = numeric_fixed_points(params, cfg["numeric_fixed_points"]["seed_grids"])
        out.append({"width": width, "r": r, "u_w": u_w, "u_h": u_h, "plateau_neg": m_n, "plateau_pos": m_p,
                    "n_reference": len(reference), "n_primary": len(primary), "agree": len(reference) == len(primary)})
    return out


def _counts(values) -> dict:
    out: dict = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: float(kv[0]) if kv[0].replace('.', '', 1).lstrip('-').isdigit() else kv[0]))


def summarize_e1(step: list[dict], smooth: list[dict], reference: list[dict], cfg: dict) -> dict:
    by_plateau, by_inertia = {}, {}
    for rec in step:
        by_plateau.setdefault(f"neg{rec['plateau_neg']}_pos{rec['plateau_pos']}", []).append(rec["n_completed"])
        by_inertia.setdefault(f"rw{rec['r_w']}_rh{rec['r_h']}", []).append(rec["n_completed"])
    five_at_zero = {}
    for rec in step:
        if rec["u_w"] == 0 and rec["u_h"] == 0 and rec["r_w"] == rec["r_h"] and rec["n_completed"] == 5:
            five_at_zero.setdefault(str(rec["r_w"]), []).append([rec["plateau_neg"], rec["plateau_pos"]])
    smooth_by_width = {}
    for rec in smooth:
        d = smooth_by_width.setdefault(str(rec["width"]), {"couples": 0, "n_fixed": [], "n_stable": [], "even": 0, "agree_step_completed": 0, "stable_minus_unstable": []})
        d["couples"] += 1; d["n_fixed"].append(rec["n_fixed"]); d["n_stable"].append(rec["n_stable"]); d["even"] += rec["even_count"]
        d["agree_step_completed"] += rec["agrees_with_step_completed"]; d["stable_minus_unstable"].append(rec["n_stable"] - rec["n_unstable"])
    for d in smooth_by_width.values():
        d["distribution_n_fixed"] = _counts(d.pop("n_fixed")); d["distribution_n_stable"] = _counts(d.pop("n_stable"))
        d["distribution_stable_minus_unstable"] = _counts(d.pop("stable_minus_unstable"))
        d["fraction_agreeing_with_step_completed_count"] = d.pop("agree_step_completed") / d["couples"]
        d["even_count_couples"] = d.pop("even")
    smooth_primary_by_plateau = {}
    for rec in smooth:
        if rec["width"] == cfg["smooth_primary_width"]:
            smooth_primary_by_plateau.setdefault(f"neg{rec['plateau_neg']}_pos{rec['plateau_pos']}", []).append(rec["n_fixed"])
    return {
        "claim": cfg["claim"],
        "step_form": {
            "couples": len(step),
            "distribution_n_fixed": _counts(r["n_fixed"] for r in step),
            "distribution_n_stable": _counts(r["n_stable"] for r in step),
            "distribution_n_completed": _counts(r["n_completed"] for r in step),
            "max_n_fixed": max(r["n_fixed"] for r in step),
            "all_strict_fixed_points_stable": all(r["n_fixed"] == r["n_stable"] for r in step),
            "completed_equals_2n_minus_1_fraction": float(np.mean([r["n_completed"] == 2 * r["n_fixed"] - 1 for r in step])),
            "solver_vs_closed_form_mismatches": int(sum(sorted(r["types"]) != sorted(r["predicted_types"]) for r in step)),
            "degenerate_couples": int(sum(r["degenerate"] for r in step)),
            "distribution_n_completed_neg_ge_pos": _counts(r["n_completed"] for r in step if r["neg_ge_pos"]),
            "by_plateau_pair": {k: _counts(v) for k, v in by_plateau.items()},
            "by_inertia_pair": {k: _counts(v) for k, v in by_inertia.items()},
            "plateau_pairs_with_five_at_u0_by_shared_inertia": five_at_zero,
            "type_frequency": _counts(t for r in step for t in r["types"]),
        },
        "smooth_form": {
            "couples": len(smooth), "by_width": smooth_by_width,
            "primary_width_by_plateau_pair": {k: _counts(v) for k, v in smooth_primary_by_plateau.items()},
            "reference_check": {"couples": len(reference), "disagreements": int(sum(not r["agree"] for r in reference)),
                                "reference_seeds": cfg["numeric_fixed_points"]["reference_seeds"]},
        },
    }


# ----------------------------------------------------------------------------- E2
def e2_sweep(cfg: dict) -> list[dict]:
    sw = cfg["plateau_sweep"]
    values = [round(float(x), 6) for x in np.arange(sw["start"], sw["stop"] + sw["step"] / 2, sw["step"])]
    records = []
    for group in cfg["groups"]:
        pub = PUBLISHED_1999[group]
        target = (pub["wife"]["influenced"], pub["husband"]["influenced"])
        for m_n in values:
            for m_p in values:
                fps = dyad.steady_states(published_couple(group, m_n, m_p))
                dists = [max(abs(s["w"] - target[0]), abs(s["h"] - target[1])) for s in fps]
                i = int(np.argmin(dists)) if dists else None
                records.append({"group": group, "plateau_neg": m_n, "plateau_pos": m_p, "n_fixed": len(fps),
                                "n_stable": int(sum(s["stable"] for s in fps)),
                                "fixed_points": [[s["w"], s["h"], s["stable"]] for s in fps],
                                "closest": [fps[i]["w"], fps[i]["h"]] if i is not None else None,
                                "closest_distance": dists[i] if i is not None else None,
                                "within_tolerance": bool(dists[i] <= cfg["match_tolerance"]) if i is not None else False,
                                "published_influenced": list(target)})
    return records


def e2_basins(cfg: dict) -> tuple[list[dict], dict]:
    g = cfg["initial_grid"]
    axis = np.arange(g["min"], g["max"] + g["step"] / 2, g["step"], dtype=float)
    ics = np.array(list(itertools.product(axis, axis)))
    rows, arrays = [], {"initial_conditions": ics}
    for group in cfg["groups"]:
        for m_n, m_p in cfg["basin_plateau_pairs"]:
            params = published_couple(group, m_n, m_p)
            fps = dyad.steady_states(params)
            finals = np.empty((len(ics), 2)); converged = np.zeros(len(ics), bool); nearest = np.full(len(ics), -1)
            for i, (w0, h0) in enumerate(ics):
                traj = dyad.simulate(params, w0, h0, cfg["steps"], clip=cfg["clip"])
                finals[i] = traj[-1]
                converged[i] = bool(np.abs(traj[-1] - traj[-2]).max() < 1e-6)
                if fps:
                    nearest[i] = int(np.argmin([abs(traj[-1][0] - s["w"]) + abs(traj[-1][1] - s["h"]) for s in fps]))
            key = f"{group}_neg{m_n:g}_pos{m_p:g}"
            arrays[key + "_finals"] = finals; arrays[key + "_nearest"] = nearest
            negative = float(np.mean((finals[:, 0] < 0) & (finals[:, 1] < 0)))
            positive = float(np.mean((finals[:, 0] > 0) & (finals[:, 1] > 0)))
            rows.append({"group": group, "plateau_neg": m_n, "plateau_pos": m_p, "initial_conditions": int(len(ics)),
                         "fixed_points": [[s["w"], s["h"], s["stable"]] for s in fps],
                         "fraction_negative_basin": negative, "fraction_positive_basin": positive,
                         "fraction_mixed": float(1 - negative - positive), "fraction_converged": float(converged.mean()),
                         "fraction_per_fixed_point": [float(np.mean(nearest == k)) for k in range(len(fps))]})
    return rows, arrays


def summarize_e2(sweep: list[dict], basins: list[dict], cfg: dict) -> dict:
    groups = {}
    for group in cfg["groups"]:
        rows = [r for r in sweep if r["group"] == group]
        pub = PUBLISHED_1999[group]
        within = [r for r in rows if r["within_tolerance"]]
        best = min(rows, key=lambda r: r["closest_distance"] if r["closest_distance"] is not None else np.inf)
        groups[group] = {
            "published_uninfluenced": [pub["wife"]["uninfluenced"], pub["husband"]["uninfluenced"]],
            "published_influenced": [pub["wife"]["influenced"], pub["husband"]["influenced"]],
            "plateau_pairs": len(rows), "pairs_within_tolerance": len(within),
            "within_tolerance_pairs": "all" if len(within) == len(rows) else [[r["plateau_neg"], r["plateau_pos"]] for r in within],
            "closest_distance_min": best["closest_distance"], "closest_point": best["closest"],
            "closest_plateau_pair": [best["plateau_neg"], best["plateau_pos"]],
            "distribution_n_fixed": _counts(r["n_fixed"] for r in rows),
            "interior_fixed_point_equals_uninfluenced": all(any(abs(fp[0] - pub["wife"]["uninfluenced"]) < 1e-9 and abs(fp[1] - pub["husband"]["uninfluenced"]) < 1e-9
                                                               for fp in r["fixed_points"]) for r in rows),
        }
    return {"tolerance": cfg["match_tolerance"], "groups": groups, "basins": basins,
            "note": "Published influenced steady states are group means of per-couple graphical estimates; the model couple built from mean parameters has the uninfluenced point as its interior fixed point whenever both uninfluenced means lie inside the thresholds, so coordinate-wise agreement within tolerance is only possible when the published influenced and uninfluenced means are themselves within tolerance."}


# ----------------------------------------------------------------------------- E3
def simulate_noisy_series(params: dict, w0: int, h0: int, steps: int, noise: np.ndarray, clip: tuple) -> np.ndarray:
    """Process noise on both equations; the rounded, clipped integer is the propagated state."""
    out = np.empty((steps + 1, 2), dtype=np.int16); out[0] = (w0, h0)
    w, h = float(w0), float(h0)
    lo, hi = clip
    for t in range(steps):
        w1, h1 = dyad.step(w, h, params)
        w = float(min(hi, max(lo, round(float(w1 + noise[t, 0])))))
        h = float(min(hi, max(lo, round(float(h1 + noise[t, 1])))))
        out[t + 1] = (int(w), int(h))
    return out


def compact_published(fit: dict) -> dict:
    out = {}
    for name in ("w", "h"):
        est = fit.get(name)
        if est is None:
            out[name] = None
            continue
        inf = est["influence"]
        out[name] = {"inertia": est["inertia"], "constant": est["constant"], "uninfluenced": est["uninfluenced"], "subset": est["subset"],
                     "neg_threshold": inf["neg_threshold"], "pos_threshold": inf["pos_threshold"],
                     "neg_value": inf["neg_value"], "pos_value": inf["pos_value"]}
    return out


def _e3_task(args) -> tuple[list[dict], np.ndarray]:
    cfg, group, sd, reps = args
    params = published_couple(group, *cfg["plateau"])
    tg = cfg["threshold_grid"]
    grid = np.arange(tg["start"], tg["stop"] + tg["step"] / 2, tg["step"], dtype=float)
    lo, hi = cfg["initial_range"]
    records, series = [], []
    for rep in reps:
        rng = np.random.default_rng(np.random.SeedSequence([cfg["base_seed"], GROUPS.index(group), cfg["noise_sd"].index(sd), rep]))
        w0, h0 = (int(v) for v in rng.integers(lo, hi + 1, 2))
        noise = rng.normal(0., sd, (cfg["series_length"] - 1, 2))
        s = simulate_noisy_series(params, w0, h0, cfg["series_length"] - 1, noise, tuple(SPAFF_RANGE))
        w, h = s[:, 0].astype(float), s[:, 1].astype(float)
        est = {"published_tol0": compact_published(dyad.fit_published_algorithm(w, h, zero_tol=cfg["zero_tol_primary"], form="ojive", threshold_grid=grid))}
        if sd in cfg["zero_tol_sensitivity_noise_sd"]:
            est["published_tol1"] = compact_published(dyad.fit_published_algorithm(w, h, zero_tol=cfg["zero_tol_sensitivity"], form="ojive", threshold_grid=grid))
        joint = dyad.fit_joint_bilinear(w, h)
        est["joint_bilinear"] = {n: {"inertia": joint[n]["inertia"], "constant": joint[n]["constant"], "uninfluenced": joint[n]["uninfluenced"],
                                     "slope_neg": joint[n]["influence"]["slope_neg"], "slope_pos": joint[n]["influence"]["slope_pos"]} for n in ("w", "h")}
        if sd in cfg["rls"]["noise_sd"]:
            est["rls"] = {}
            for me, other, name in ((w, h, "w"), (h, w, "h")):
                ident = dyad.RecursiveIdentifier(lam=cfg["rls"]["lam"], prior=cfg["rls"]["prior"])
                for t in range(len(me) - 1):
                    ident.update(me[t], other[t], me[t + 1])
                p = ident.params()
                est["rls"][name] = {"inertia": p["inertia"], "constant": p["constant"], "uninfluenced": p["uninfluenced"],
                                    "slope_neg": p["slope_neg"], "slope_pos": p["slope_pos"]}
        records.append({"group": group, "noise_sd": sd, "rep": rep, "w0": w0, "h0": h0,
                        "zero_count_h": int(np.sum(h[:-1] == 0)), "zero_count_w": int(np.sum(w[:-1] == 0)), "estimates": est})
        series.append(s)
    return records, np.stack(series)


def truth_for(params: dict, spouse: str) -> dict:
    r, c, spec = (params["r1"], params["a"], params["I_hw"]) if spouse == "w" else (params["r2"], params["b"], params["I_wh"])
    return {"inertia": r, "uninfluenced": c / (1 - r), "neg_threshold": spec["neg_threshold"], "pos_threshold": spec["pos_threshold"],
            "neg_value": spec["neg_value"], "pos_value": spec["pos_value"]}


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> list[float] | None:
    if len(values) < 2:
        return None
    boot = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def summarize_e3(records: list[dict], cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    rows, zeros = [], []
    for group in cfg["groups"]:
        params = published_couple(group, *cfg["plateau"])
        for sd in cfg["noise_sd"]:
            subset = [r for r in records if r["group"] == group and r["noise_sd"] == sd]
            zeros.append({"group": group, "noise_sd": sd, "mean_partner_zero_count_for_w": float(np.mean([r["zero_count_h"] for r in subset])),
                          "mean_partner_zero_count_for_h": float(np.mean([r["zero_count_w"] for r in subset]))})
            estimators = [k for k in ("published_tol0", "published_tol1", "joint_bilinear", "rls") if k in subset[0]["estimates"]]
            for est in estimators:
                for spouse in ("w", "h"):
                    truth = truth_for(params, spouse)
                    values = [r["estimates"][est][spouse] for r in subset]
                    valid = [v for v in values if v is not None]
                    row = {"group": group, "noise_sd": sd, "estimator": est, "spouse": spouse, "replications": len(subset),
                           "fraction_none": float(np.mean([v is None for v in values])),
                           "mean_zero_subset": float(np.mean([v["subset"] for v in valid])) if valid and "subset" in valid[0] else None,
                           "parameters": {}}
                    for p in ESTIMATED:
                        if not valid or p not in valid[0]:
                            continue
                        x = np.array([v[p] if v[p] is not None else np.nan for v in valid], float)
                        x = x[np.isfinite(x)]
                        err = x - truth[p]
                        row["parameters"][p] = {"truth": truth[p], "n": int(len(x)),
                                                "bias": float(err.mean()) if len(x) else None,
                                                "bias_ci95": bootstrap_mean_ci(err, rng, cfg["bootstrap"]["samples"]),
                                                "rmse": float(np.sqrt(np.mean(err ** 2))) if len(x) else None,
                                                "median_abs_error": float(np.median(np.abs(err))) if len(x) else None,
                                                "sd": float(err.std(ddof=1)) if len(x) > 1 else None}
                    rows.append(row)
    return {"plateau": cfg["plateau"], "replications_per_cell": cfg["replications"], "partner_zero_counts": zeros, "cells": rows,
            "note": "Bias and RMSE exclude replications where the zero subset had fewer than three points (fraction_none) and non-finite estimates. The comparison estimators fit a bilinear partner equation to ojive-generated data, so only inertia and uninfluenced steady state are comparable for them."}


# ----------------------------------------------------------------------------- E4
def quadrant(w: float, h: float) -> str:
    return ("+" if w > 0 else "-") + ("+" if h > 0 else "-")


def cycle_check(params: dict, cfg: dict) -> dict:
    u = dyad.uninfluenced_steady_states(params)
    traj = dyad.simulate(params, u["w"], u["h"], cfg["burn_in"] + cfg["window"])
    window = traj[cfg["burn_in"]:]
    bounded = bool(np.isfinite(window).all() and np.abs(window).max() < 1e3)
    period = None
    if bounded:
        for p in range(1, cfg["max_period"] + 1):
            if np.abs(window[p:] - window[:-p]).max() < 1e-6:
                period = p
                break
    return {"period": period, "bounded": bounded,
            "w_range": [float(window[:, 0].min()), float(window[:, 0].max())] if bounded else None,
            "h_range": [float(window[:, 1].min()), float(window[:, 1].max())] if bounded else None}


def e4_records(cfg: dict) -> list[dict]:
    records = []
    repairs = [None] + [(e, k) for e in cfg["repair_effectiveness"] for k in cfg["repair_thresholds"]]
    dampings = [None] + [(e, k) for e in cfg["damping_effectiveness"] for k in cfg["damping_thresholds"]]
    for group in cfg["groups"]:
        for slopes in cfg["slopes"]:
            for repair in repairs:
                for damping in dampings:
                    params = repair_damping_couple(group, slopes, repair, damping)
                    fps = dyad.steady_states(params)
                    stable = [s for s in fps if s["stable"]]
                    condition = "none" if repair is None and damping is None else "repair" if damping is None else "damping" if repair is None else "both"
                    rec = {"group": group, "slope_neg": slopes[0], "slope_pos": slopes[1], "condition": condition,
                           "repair_effectiveness": repair[0] if repair else None, "repair_threshold": repair[1] if repair else None,
                           "damping_effectiveness": damping[0] if damping else None, "damping_threshold": damping[1] if damping else None,
                           "n_fixed": len(fps), "n_stable": len(stable),
                           "stable_per_quadrant": {q: int(sum(quadrant(s["w"], s["h"]) == q for s in stable)) for q in QUADRANTS},
                           "fixed_points": [[s["w"], s["h"], s["stable"], s["spectral_radius"]] for s in fps]}
                    if not fps:
                        rec["cycle"] = cycle_check(params, cfg["cycle_check"])
                    records.append(rec)
    return records


def summarize_e4(records: list[dict], cfg: dict) -> dict:
    def key(r, with_repair=True, with_damping=True):
        rep = (r["repair_effectiveness"], r["repair_threshold"]) if with_repair and r["repair_effectiveness"] is not None else None
        dmp = (r["damping_effectiveness"], r["damping_threshold"]) if with_damping and r["damping_effectiveness"] is not None else None
        return (r["group"], r["slope_neg"], r["slope_pos"], rep, dmp)
    index = {key(r): r for r in records}
    created_by_damping, destroyed_by_damping, created_by_repair, destroyed_by_repair = [], [], [], []
    for r in records:
        if r["damping_effectiveness"] is not None:
            base = index[key(r, with_damping=False)]
            before, after = base["stable_per_quadrant"]["++"], r["stable_per_quadrant"]["++"]
            entry = {k: r[k] for k in ("group", "slope_neg", "slope_pos", "condition", "repair_effectiveness", "repair_threshold", "damping_effectiveness", "damping_threshold")}
            entry.update(pp_stable_before=before, pp_stable_after=after)
            if before == 0 and after > 0:
                created_by_damping.append(entry)
            if before > 0 and after == 0:
                destroyed_by_damping.append(entry)
        if r["repair_effectiveness"] is not None:
            base = index[key(r, with_repair=False)]
            before, after = base["stable_per_quadrant"]["--"], r["stable_per_quadrant"]["--"]
            entry = {k: r[k] for k in ("group", "slope_neg", "slope_pos", "condition", "repair_effectiveness", "repair_threshold", "damping_effectiveness", "damping_threshold")}
            entry.update(nn_stable_before=before, nn_stable_after=after)
            if before == 0 and after > 0:
                created_by_repair.append(entry)
            if before > 0 and after == 0:
                destroyed_by_repair.append(entry)
    table = {}
    for r in records:
        d = table.setdefault(f"{r['group']}|slopes{r['slope_neg']}/{r['slope_pos']}|{r['condition']}", {"configurations": 0, "mean_stable_per_quadrant": {q: 0. for q in QUADRANTS}, "fraction_with_pp_stable": 0., "fraction_with_no_fixed_point": 0.})
        d["configurations"] += 1
        for q in QUADRANTS:
            d["mean_stable_per_quadrant"][q] += r["stable_per_quadrant"][q]
        d["fraction_with_pp_stable"] += r["stable_per_quadrant"]["++"] > 0
        d["fraction_with_no_fixed_point"] += r["n_fixed"] == 0
    for d in table.values():
        n = d["configurations"]
        d["mean_stable_per_quadrant"] = {q: v / n for q, v in d["mean_stable_per_quadrant"].items()}
        d["fraction_with_pp_stable"] /= n; d["fraction_with_no_fixed_point"] /= n
    none_rows = [r for r in records if r["condition"] == "none"]
    return {"claim": "damping may create a positive-positive stable steady state where none existed before",
            "configurations": len(records), "table": table,
            "baseline_pp_stable_by_group_and_slopes": {f"{r['group']}|slopes{r['slope_neg']}/{r['slope_pos']}": r["stable_per_quadrant"]["++"] for r in none_rows},
            "baseline_fixed_points": {f"{r['group']}|slopes{r['slope_neg']}/{r['slope_pos']}": r["fixed_points"] for r in none_rows},
            "damping_configurations": int(sum(r["damping_effectiveness"] is not None for r in records)),
            "repair_configurations": int(sum(r["repair_effectiveness"] is not None for r in records)),
            "n_positive_positive_created_by_damping": len(created_by_damping), "n_positive_positive_destroyed_by_damping": len(destroyed_by_damping),
            "n_negative_negative_created_by_repair": len(created_by_repair), "n_negative_negative_destroyed_by_repair": len(destroyed_by_repair),
            "n_configurations_without_fixed_point": int(sum(r["n_fixed"] == 0 for r in records)),
            "positive_positive_created_by_damping": created_by_damping, "positive_positive_destroyed_by_damping": destroyed_by_damping,
            "negative_negative_created_by_repair": created_by_repair, "negative_negative_destroyed_by_repair": destroyed_by_repair,
            "configurations_without_fixed_point": [{k: r[k] for k in ("group", "slope_neg", "slope_pos", "condition", "repair_effectiveness", "repair_threshold", "damping_effectiveness", "damping_threshold", "cycle")} for r in records if r["n_fixed"] == 0],
            "unstable_fixed_point_configurations": int(sum(r["n_fixed"] != r["n_stable"] for r in records))}


# ----------------------------------------------------------------------------- driver
def chunks(seq: list, size: int) -> list[list]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def parallel_map(fn, tasks: list, workers: int) -> list:
    if workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, tasks))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def run(config_path: str | Path, output: str | Path, workers: int | None = None) -> dict:
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; preserve prior measurements")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "source.py").write_bytes(Path(__file__).read_bytes())
    (output / "dyad_source.py").write_bytes(Path(dyad.__file__).read_bytes())
    workers = int(config.get("workers", 1)) if workers is None else int(workers)
    start = time.perf_counter()
    timings = {}

    # E1 structural
    t0 = time.perf_counter()
    e1 = config["e1_structural"]
    step_records = e1_step_records(e1)
    combos = smooth_combos(e1)
    smooth_records = [r for batch in parallel_map(_e1_smooth_task, [(e1, c) for c in chunks(combos, 24)], workers) for r in batch]
    ref_rng = np.random.default_rng(e1["numeric_fixed_points"]["reference_seed"])
    sample = [combos[i] for i in sorted(ref_rng.choice(len(combos), size=min(e1["numeric_fixed_points"]["reference_sample"], len(combos)), replace=False))]
    reference = [r for batch in parallel_map(_e1_reference_task, [(e1, c) for c in chunks(sample, 4)], workers) for r in batch]
    timings["e1_seconds"] = time.perf_counter() - t0
    write_jsonl(output / "e1_structural.jsonl.gz", step_records + smooth_records)
    write_jsonl(output / "e1_reference_check.jsonl.gz", reference)
    print(f"E1 done: {len(step_records)} step couples, {len(smooth_records)} smooth couples ({timings['e1_seconds']:.1f}s)", flush=True)

    # E2 published parameters
    t0 = time.perf_counter()
    e2 = config["e2_published"]
    sweep = e2_sweep(e2)
    basin_rows, basin_arrays = e2_basins(e2)
    timings["e2_seconds"] = time.perf_counter() - t0
    write_jsonl(output / "e2_plateau_sweep.jsonl.gz", sweep)
    np.savez_compressed(output / "e2_basins.npz", **basin_arrays)
    print(f"E2 done: {len(sweep)} sweep rows, {len(basin_rows)} basin maps ({timings['e2_seconds']:.1f}s)", flush=True)

    # E3 estimation recovery
    t0 = time.perf_counter()
    e3 = {**config["e3_recovery"], "series_length": config["series_length"]}
    tasks = [(e3, group, sd, reps) for group in e3["groups"] for sd in e3["noise_sd"] for reps in chunks(list(range(e3["replications"])), 50)]
    results = parallel_map(_e3_task, tasks, workers)
    e3_records = [r for recs, _ in results for r in recs]
    series = np.concatenate([arr for _, arr in results])
    index = np.array([[GROUPS.index(r["group"]), e3["noise_sd"].index(r["noise_sd"]), r["rep"]] for r in e3_records], dtype=np.int32)
    timings["e3_seconds"] = time.perf_counter() - t0
    write_jsonl(output / "e3_recovery.jsonl.gz", e3_records)
    np.savez_compressed(output / "e3_series.npz", series=series, index=index, groups=np.array(GROUPS), noise_sd=np.array(e3["noise_sd"]))
    print(f"E3 done: {len(e3_records)} simulated series ({timings['e3_seconds']:.1f}s)", flush=True)

    # E4 repair and damping
    t0 = time.perf_counter()
    e4 = config["e4_repair_damping"]
    e4_recs = e4_records(e4)
    timings["e4_seconds"] = time.perf_counter() - t0
    write_jsonl(output / "e4_repair_damping.jsonl.gz", e4_recs)
    print(f"E4 done: {len(e4_recs)} configurations ({timings['e4_seconds']:.1f}s)", flush=True)

    summary = {
        "papers": config["papers"], "protocol_frozen_utc": config["protocol_frozen_utc"],
        "fidelity": "computational reproduction of model properties and estimation procedure from published group-mean parameters; no couple data; model code imported from regesh.bonding.dyad",
        "e1_structural": summarize_e1(step_records, smooth_records, reference, e1),
        "e2_published": summarize_e2(sweep, basin_rows, e2),
        "e3_recovery": summarize_e3(e3_records, e3),
        "e4_repair_damping": summarize_e4(e4_recs, e4),
        "timings": timings, "runtime_seconds": time.perf_counter() - start, "workers": workers,
        "python": platform.python_version(), "numpy": np.__version__,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {"schema_version": 1,
                "papers": [{"doi": p["doi"], "url": f"https://doi.org/{p['doi']}"} for p in config["papers"]],
                "fidelity": summary["fidelity"], "author_code_executed": False, "author_code_copied": False, "couple_data_used": False,
                "model_module": "regesh/bonding/dyad.py", "model_module_sha256": hashlib.sha256(Path(dyad.__file__).read_bytes()).hexdigest(),
                "python": platform.python_version(), "numpy": np.__version__, "workers": workers,
                "command": f"python -m regesh.reproductions.marital_dynamics --config {config_path} --output {output}",
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reproductions/marital_dynamics.json")
    parser.add_argument("--output", required=True, help="new, empty directory")
    parser.add_argument("--workers", type=int, default=None, help="process pool size; results do not depend on it")
    args = parser.parse_args()
    run(args.config, args.output, args.workers)


if __name__ == "__main__":
    main()
