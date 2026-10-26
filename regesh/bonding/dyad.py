"""Dyadic affect dynamics: coupled difference equations with influence functions.

Model of Cook et al. (1995), Gottman, Swanson & Murray (1999, J. Family
Psychology 13:3-19) and Gottman et al. (2003, J. Homosexuality 45:65-91):

    W_{t+1} = a + r1 W_t + I_HW(H_t) [+ E_R R(H_t; K_R) + E_D D(H_t; K_D)]
    H_{t+1} = b + r2 H_t + I_WH(W_t) [+ ...]

Influence functions are ojive (1999, Figure 1A) or bilinear (2003, Figure 1).
Repair is a trigger that adds amplitude E once the partner's score falls to or
below threshold K; damping mirrors it above a positive threshold with a
negative amplitude. Uninfluenced steady states are a/(1-r1) and b/(1-r2).

Also provided: the published estimation algorithm (fit the autoregressive part on
the subset where the partner scores zero, then fit an influence form to the
residuals), a joint least-squares alternative, and an online recursive
least-squares identifier of a partner's equation for use by an agent that
chooses its own expressed affect. Steering recommendations are bounded so the
agent never simply mirrors the partner (mid-range coordination; Jaffe et al.
2001). Nothing here estimates intent or subjective feeling.
"""
from __future__ import annotations
import itertools
import math
import numpy as np

# Gottman, Swanson & Murray (1999) Table 1, group means (N=108 stable, 17 divorced),
# weighted SPAFF in 6-s blocks, range -24..+24, 150 points per spouse.
PUBLISHED_1999 = {
    "stable": {"wife": {"inertia": .39, "uninfluenced": .51, "pos_threshold": 8.25, "neg_threshold": -6.23, "influenced": -.65},
               "husband": {"inertia": .40, "uninfluenced": .72, "pos_threshold": 8.45, "neg_threshold": -5.36, "influenced": .71}},
    "divorced": {"wife": {"inertia": .42, "uninfluenced": -2.26, "pos_threshold": 7.24, "neg_threshold": -6.06, "influenced": -1.95},
                 "husband": {"inertia": .30, "uninfluenced": -1.89, "pos_threshold": 6.65, "neg_threshold": -7.71, "influenced": -1.49}},
    # Table 2 three-group means (df 2,48 / 2,41)
    "happy_stable": {"wife": {"inertia": .37, "uninfluenced": 1.07, "pos_threshold": 9.12, "neg_threshold": -5.76, "influenced": -.05},
                     "husband": {"inertia": .38, "uninfluenced": 1.06, "pos_threshold": 7.94, "neg_threshold": -5.94, "influenced": -.33}},
    "unhappy_stable": {"wife": {"inertia": .41, "uninfluenced": -.03, "pos_threshold": 8.53, "neg_threshold": -6.76, "influenced": -1.88},
                       "husband": {"inertia": .35, "uninfluenced": -.41, "pos_threshold": 7.94, "neg_threshold": -5.29, "influenced": .31}},
}
SPAFF_RANGE = (-24, 24)

# Pooled least-squares bilinear fit of partner next-affect on a hash-selected 20% of
# DailyDialog training dialogues (speaker A as partner, speaker B as the agent role,
# declared seven-emotion valence map), from results/bonding/partner_model/summary.json.
# Order: constant, inertia, slope_neg, slope_pos. Human-human conversation data; the
# positive slope exceeding the negative one is corpus-specific and the reverse of the
# asymmetry Gottman reports for marital conflict.
DAILYDIALOG_POOLED_PRIOR = (0.03691770901755599, 0.3546044474577197, 0.12021559885929885, 0.34968725826500563)


def ojive(x, neg_threshold, neg_value, pos_threshold, pos_value):
    """1999 Figure 1A: no influence inside the thresholds, fixed values outside."""
    if x <= neg_threshold:
        return neg_value
    if x >= pos_threshold:
        return pos_value
    return 0.


def bilinear(x, slope_neg, slope_pos):
    """2003 Figure 1: separate slopes for negative and positive partner affect; zero at zero."""
    return slope_neg * x if x < 0 else slope_pos * x


def repair(x, threshold, effectiveness):
    """Trigger adding amplitude E once the partner's score is at or below negativity threshold K."""
    return effectiveness if x <= threshold else 0.


def damping(x, threshold, effectiveness):
    """Mirror of repair above a positivity threshold; effectiveness is expected to be negative."""
    return effectiveness if x >= threshold else 0.


def sigmoid_ojive(x, neg_threshold, neg_value, pos_threshold, pos_value, width):
    """Smooth two-sided sigmoid with zero influence at zero.

    The 1999 paper calls its form "O-Jive or sigmoidal"; the step version (`ojive`)
    admits at most three fixed points because each nullcline is three disjoint
    segments, so the published "one, three, or five steady states" requires a
    smooth transition such as this one (logistic ramps centred on the thresholds).
    """
    if x > 0:
        return pos_value / (1 + math.exp(-(x - pos_threshold) / width)) - pos_value / (1 + math.exp(pos_threshold / width))
    if x < 0:
        return neg_value / (1 + math.exp((x - neg_threshold) / width)) - neg_value / (1 + math.exp(-neg_threshold / width))
    return 0.


def influence(x, spec):
    """Evaluate an influence specification: {'form': 'ojive'|'sigmoid'|'bilinear'|'none', ...} plus optional repair/damping."""
    form = spec.get("form", "bilinear")
    if form == "ojive":
        value = ojive(x, spec["neg_threshold"], spec["neg_value"], spec["pos_threshold"], spec["pos_value"])
    elif form == "sigmoid":
        value = sigmoid_ojive(x, spec["neg_threshold"], spec["neg_value"], spec["pos_threshold"], spec["pos_value"], spec.get("width", 1.))
    elif form == "bilinear":
        value = bilinear(x, spec["slope_neg"], spec["slope_pos"])
    elif form == "none":
        value = 0.
    else:
        raise ValueError("unknown influence form")
    if "repair" in spec:
        value += repair(x, spec["repair"]["threshold"], spec["repair"]["effectiveness"])
    if "damping" in spec:
        value += damping(x, spec["damping"]["threshold"], spec["damping"]["effectiveness"])
    return value


def step(w, h, params):
    """One iteration of the coupled map. params: {'a','r1','b','r2','I_hw':spec,'I_wh':spec}."""
    w1 = params["a"] + params["r1"] * w + influence(h, params["I_hw"])
    h1 = params["b"] + params["r2"] * h + influence(w, params["I_wh"])
    return w1, h1


def simulate(params, w0, h0, steps, clip=None):
    w, h = float(w0), float(h0)
    out = np.empty((steps + 1, 2)); out[0] = (w, h)
    for t in range(1, steps + 1):
        w, h = step(w, h, params)
        if clip is not None:
            w, h = max(clip[0], min(clip[1], w)), max(clip[0], min(clip[1], h))
        out[t] = (w, h)
    return out


def uninfluenced_steady_states(params):
    return {"w": params["a"] / (1 - params["r1"]), "h": params["b"] / (1 - params["r2"])}


def _pieces(spec):
    """Enumerate linear pieces (interval, slope, intercept) of a piecewise-linear influence spec."""
    form = spec.get("form", "bilinear")
    cuts = set()
    if form == "ojive":
        cuts |= {spec["neg_threshold"], spec["pos_threshold"]}
    elif form == "bilinear":
        cuts.add(0.)
    if "repair" in spec: cuts.add(spec["repair"]["threshold"])
    if "damping" in spec: cuts.add(spec["damping"]["threshold"])
    bounds = [-math.inf, *sorted(cuts), math.inf]
    pieces = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        probe = 0. if lo == -math.inf and hi == math.inf else (lo + 1e-9 if hi == math.inf else (hi - 1e-9 if lo == -math.inf else (lo + hi) / 2))
        # slope from two probes inside the open interval
        eps = 1e-6
        x1, x2 = probe - eps, probe + eps
        if lo != -math.inf: x1 = max(x1, lo + 1e-9)
        if hi != math.inf: x2 = min(x2, hi - 1e-9)
        y1, y2 = influence(x1, spec), influence(x2, spec)
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.
        pieces.append((lo, hi, slope, y1 - slope * x1))
    return pieces


def fixed_points_numeric(params, box=(-24., 24.), seeds=25, tol=1e-9, iterations=100):
    """Fixed points of a general (possibly smooth) map by damped Newton from a grid of seeds.

    Works for every influence form; for piecewise-linear forms prefer `steady_states`,
    which is exact. Stability from the numerical Jacobian's spectral radius.
    """
    def F(v):
        w1, h1 = step(v[0], v[1], params)
        return np.array([w1 - v[0], h1 - v[1]])
    def J(v, eps=1e-6):
        cols = []
        for i in range(2):
            d = np.zeros(2); d[i] = eps
            cols.append((F(v + d) - F(v - d)) / (2 * eps))
        return np.column_stack(cols)
    found = []
    grid = np.linspace(box[0], box[1], seeds)
    for w0 in grid:
        for h0 in grid:
            v = np.array([w0, h0], float)
            for _ in range(iterations):
                f = F(v)
                if np.abs(f).max() < tol:
                    break
                try:
                    dv = np.linalg.solve(J(v), -f)
                except np.linalg.LinAlgError:
                    break
                scale = min(1., 4. / (np.abs(dv).max() + 1e-12))
                v = v + scale * dv
                if not np.all(np.isfinite(v)) or np.abs(v).max() > 1e3:
                    break
            if np.all(np.isfinite(v)) and np.abs(F(v)).max() < 1e-7 and box[0] - 1 <= v.min() and v.max() <= box[1] + 1:
                if not any(abs(v[0] - s["w"]) < 1e-5 and abs(v[1] - s["h"]) < 1e-5 for s in found):
                    jac = J(v) + np.eye(2)
                    rho = float(max(abs(np.linalg.eigvals(jac))))
                    found.append({"w": float(v[0]), "h": float(v[1]), "stable": bool(rho < 1), "spectral_radius": rho})
    found.sort(key=lambda s: (s["w"], s["h"]))
    return found


def steady_states(params, tol=1e-9):
    """All fixed points of the piecewise-linear map, with linear stability.

    With step (ojive) influence each nullcline is three disjoint axis-parallel
    segments with disjoint ranges, so at most three fixed points exist. Smooth
    'sigmoid' forms are not piecewise-linear; use `fixed_points_numeric` for them.

    Each combination of a piece of I_hw (in h) and a piece of I_wh (in w) gives a
    linear system; a solution is kept when it lies inside both pieces. Boundary
    points, where the ojive jumps, are reported as unstable only if an iterate
    could rest there, which for a jump cannot happen; they are excluded.
    """
    if params["I_hw"].get("form") == "sigmoid" or params["I_wh"].get("form") == "sigmoid":
        return fixed_points_numeric(params)
    found = []
    for (hlo, hhi, sh, ch), (wlo, whi, sw, cw) in itertools.product(_pieces(params["I_hw"]), _pieces(params["I_wh"])):
        # w = a + r1 w + sh h + ch ; h = b + r2 h + sw w + cw
        A = np.array([[1 - params["r1"], -sh], [-sw, 1 - params["r2"]]])
        rhs = np.array([params["a"] + ch, params["b"] + cw])
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        w, h = np.linalg.solve(A, rhs)
        if not (hlo - tol <= h < hhi + tol and wlo - tol <= w < whi + tol):
            continue
        # exclude points sitting exactly on an open boundary of a jump
        if any(abs(h - c) <= tol for c in (hlo, hhi) if math.isfinite(c)) or any(abs(w - c) <= tol for c in (wlo, whi) if math.isfinite(c)):
            continue
        jac = np.array([[params["r1"], sh], [sw, params["r2"]]])
        rho = max(abs(np.linalg.eigvals(jac)))
        found.append({"w": float(w), "h": float(h), "stable": bool(rho < 1), "spectral_radius": float(rho)})
    found.sort(key=lambda s: (s["w"], s["h"]))
    return found


def fit_published_algorithm(w, h, zero_tol=0., form="ojive", threshold_grid=None):
    """Gottman, Swanson & Murray (1999) 'Computational Algorithms'.

    Fit W_{t+1} = r1 W_t + a on the subset of t with H_t == 0 (within zero_tol), and
    H_{t+1} = r2 H_t + b where W_t == 0. Influence values are the residuals
    W_{t+1} - r1 W_t - a plotted against H_t; an ojive (thresholds and plateau
    values) or bilinear form is fitted by least squares. Returns None entries when
    a subset is too small to fit, which is itself a measured limitation.
    """
    w = np.asarray(w, float); h = np.asarray(h, float)
    out = {"n": int(len(w) - 1)}
    for me, other, name in ((w, h, "w"), (h, w, "h")):
        x, y, z = me[:-1], me[1:], other[:-1]
        mask = np.abs(z) <= zero_tol
        if mask.sum() < 3:
            out[name] = None
            continue
        X = np.column_stack([x[mask], np.ones(mask.sum())])
        (r, c), *_ = np.linalg.lstsq(X, y[mask], rcond=None)
        resid = y - (r * x + c)
        spec = _fit_influence(z, resid, form, threshold_grid)
        out[name] = {"inertia": float(r), "constant": float(c), "uninfluenced": float(c / (1 - r)) if abs(1 - r) > 1e-9 else None,
                     "subset": int(mask.sum()), "influence": spec}
    return out


def _fit_influence(x, y, form, threshold_grid):
    if form == "bilinear":
        X = np.column_stack([np.minimum(x, 0), np.maximum(x, 0)])
        (sn, sp), *_ = np.linalg.lstsq(X, y, rcond=None)
        return {"form": "bilinear", "slope_neg": float(sn), "slope_pos": float(sp)}
    grid = threshold_grid if threshold_grid is not None else np.arange(1, 24.5, .5)
    best = None
    for kn in -grid:
        for kp in grid:
            neg, pos = x <= kn, x >= kp
            vn = y[neg].mean() if neg.any() else 0.
            vp = y[pos].mean() if pos.any() else 0.
            pred = np.where(neg, vn, np.where(pos, vp, 0.))
            sse = float(((y - pred) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, {"form": "ojive", "neg_threshold": float(kn), "neg_value": float(vn),
                              "pos_threshold": float(kp), "pos_value": float(vp), "sse": sse})
    return best[1]


def fit_joint_bilinear(w, h):
    """Alternative joint least squares: regress W_{t+1} on [1, W_t, min(H_t,0), max(H_t,0)] using all points."""
    w = np.asarray(w, float); h = np.asarray(h, float)
    out = {}
    for me, other, name in ((w, h, "w"), (h, w, "h")):
        X = np.column_stack([np.ones(len(me) - 1), me[:-1], np.minimum(other[:-1], 0), np.maximum(other[:-1], 0)])
        (c, r, sn, sp), *_ = np.linalg.lstsq(X, me[1:], rcond=None)
        out[name] = {"inertia": float(r), "constant": float(c), "uninfluenced": float(c / (1 - r)) if abs(1 - r) > 1e-9 else None,
                     "influence": {"form": "bilinear", "slope_neg": float(sn), "slope_pos": float(sp)}}
    return out


class RecursiveIdentifier:
    """Online recursive least squares for a partner's equation.

        P_{t+1} = b + r P_t + s_neg * min(A_t, 0) + s_pos * max(A_t, 0)

    where P is the partner's expressed affect and A the agent's. Forgetting factor
    lam in (0, 1] tracks slow drift. The state is a plain dict for journaling.
    """
    NAMES = ("constant", "inertia", "slope_neg", "slope_pos")

    def __init__(self, state=None, lam=.98, prior=1e2, prior_theta=None):
        """`prior` scales the initial covariance; `prior_theta` warm-starts the parameters.

        The diffuse default (P = 100 I) makes the first update a near-full jump, which the
        DailyDialog evaluation (results/bonding/partner_model) showed never recovers within
        short dialogues. Warm-starting from `DAILYDIALOG_POOLED_PRIOR` with P = I lowered
        next-affect RMSE from .275 to .244 on held-out dialogues.
        """
        if state is None:
            theta = list(prior_theta) if prior_theta is not None else [0., .3, .2, .2]
            state = {"theta": [float(v) for v in theta], "P": (np.eye(4) * prior).tolist(), "n": 0, "lam": lam, "sse": 0.}
        self.state = state

    @staticmethod
    def features(p_prev, a_prev):
        return np.array([1., p_prev, min(a_prev, 0.), max(a_prev, 0.)])

    def update(self, p_prev, a_prev, p_next):
        x = self.features(p_prev, a_prev)
        theta = np.asarray(self.state["theta"]); P = np.asarray(self.state["P"]); lam = self.state["lam"]
        err = p_next - x @ theta
        k = P @ x / (lam + x @ P @ x)
        theta = theta + k * err
        P = (P - np.outer(k, x @ P)) / lam
        P = (P + P.T) / 2
        self.state.update(theta=theta.tolist(), P=P.tolist(), n=self.state["n"] + 1, sse=self.state["sse"] + float(err ** 2))
        return float(err)

    def params(self):
        theta = dict(zip(self.NAMES, self.state["theta"]))
        r = theta["inertia"]
        theta["uninfluenced"] = theta["constant"] / (1 - r) if abs(1 - r) > 1e-9 else None
        theta["n"] = self.state["n"]
        theta["uncertainty"] = float(np.trace(np.asarray(self.state["P"])) / 4)
        return theta

    def predict(self, p_now, a_now):
        return float(self.features(p_now, a_now) @ np.asarray(self.state["theta"]))


def steer(identifier, p_now, own_valence, *, target=.5, cap=.5, repair_threshold=-.4, repair_amplitude=.3,
          min_samples=6, grid=41):
    """Choose the agent's expressed valence to move the partner toward a positive stable state.

    The adjustment from the agent's own valence is capped (|A - own| <= cap) so the
    agent cannot fully mirror or overaccommodate the partner. Repair adds a bounded
    positive jump once the partner is at or below the negativity threshold, following
    the repair term of Gottman et al. (2003). Before `min_samples` observations the
    model is not trusted and only repair applies. Returns a dict for journaling.
    """
    own = max(-1., min(1., own_valence))
    base = own
    reason = "own valence"
    trusted = identifier.state["n"] >= min_samples
    if trusted:
        candidates = np.linspace(max(-1., own - cap), min(1., own + cap), grid)
        preds = np.array([identifier.predict(p_now, a) for a in candidates])
        base = float(candidates[int(np.argmin(np.abs(preds - target) + .05 * np.abs(candidates - own)))])
        reason = "one-step model-predictive steering toward target"
    repaired = p_now <= repair_threshold
    expressed = max(-1., min(1., base + (repair_amplitude if repaired else 0.)))
    expressed = max(-1., min(1., own + max(-cap, min(cap, expressed - own))))
    closed = closed_loop_fixed_point(identifier, lambda p: expressed) if trusted else None
    return {"expressed_valence": float(expressed), "own_valence": own, "model_trusted": trusted,
            "repair": bool(repaired), "reason": reason,
            "predicted_partner_next": identifier.predict(p_now, expressed) if trusted else None,
            "closed_loop_fixed_point": closed}


def closed_loop_fixed_point(identifier, policy, iterations=200):
    """Fixed point of p -> b + r p + I(policy(p)) by iteration, with a simple stability flag."""
    p = 0.
    for _ in range(iterations):
        p_next = max(-1., min(1., identifier.predict(p, policy(p))))
        if abs(p_next - p) < 1e-9:
            break
        p = p_next
    r = identifier.params()["inertia"]
    return {"partner_affect": float(p), "stable": bool(abs(r) < 1)}
