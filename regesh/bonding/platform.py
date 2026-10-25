"""Bridge between the bonding mechanisms and the persistent agent state.

Everything here is deterministic, JSON-serialisable and replayable. Bond state is
opt-in (config `bonding: True`); without it the reference dynamics are untouched.

Cross-links from bond state into the rest of the agent are deliberately few and
listed here, because they are the mechanism and also the safety boundary:

1. Secure-base threat buffering: the raw goal-loss threat of a new occurrence is
   multiplied by (1 - gain * security * presence) of the most supportive present
   partner (Coan, Schaefer & Davidson 2006; social baseline theory). Security is
   measured responsiveness times familiarity; strangers buffer little.
2. Arousal-band regulation of exploration (Hiolle et al. 2014): high arousal
   recommends a support request and dampens exploration; low arousal favours it.
3. Steering, coordination and reciprocity outputs are recommendations in the
   actuation context for a host to use when it speaks.

The bond never changes factual reliability, authority, epistemic uncertainty or
tool permissions, and a support request is bounded by the partner's attention
budget so help-seeking cannot become a disengagement deterrent (De Freitas 2026).

Event-level semantics differ from the robot's per-timestep loop in one declared
way: responsiveness (Eq. 5) is updated once per resolved request (answered by
the partner's next occurrence with support, or unanswered on that occurrence or
after `request_window`). The independent reproduction showed that the literal
per-step reading drives responsiveness down under a responsive caregiver because
latency steps outnumber comfort steps; the per-resolution update avoids that
artefact while keeping the published rates and ranges.
"""
from __future__ import annotations
from copy import deepcopy
import math
from ..appraisal import clip
from .attachment import ADAPTIVE, arousal_step, comfort_step, responsiveness_step, profile_from_responsiveness, behavior
from .dyad import RecursiveIdentifier, steer
from .synchrony import CoordinationCounts, proportions, lsm_from_proportions, accommodation_assessment

BOND_CONFIG = {
    "bonding": True, "partner": None,
    "buffer_gain": .5, "presence_tau": 2., "bond_tau": 6.,
    "attention_cost": .1, "attention_recovery": .05, "request_window": 2.,
    "coordination_cap": .5, "target_partner_affect": .5,
    "repair_threshold": -.4, "repair_amplitude": .3, "steer_min_samples": 6,
    "profile": dict(ADAPTIVE),
    # Laurenceau, Feldman Barrett & Pietromonaco (1998) Study 2 standardized paths:
    # self-disclosure .22, partner disclosure .32, perceived responsiveness .36 -> intimacy;
    # emotional self-disclosure .37 versus factual -.01.
    "intimacy_weights": {"self": .22, "partner": .32, "responsiveness": .36}, "intimacy_rate": .1,
    "disclosure_escalation": .1, "text_cap": 4000,
}
ATTACHMENT_INITIAL = {"arousal": 0., "comfort": 0., "pending": None, "pending_since": None, "requests": 0, "answered": 0}


def enabled(state):
    return bool(state["config"].get("bonding")) and "bonds" in state


def bond_initial(cfg):
    p = cfg["profile"]
    return {"responsiveness": p["resp0"], "profile": profile_from_responsiveness(p["resp0"], p["beta_h"], p["tau_h"], p["r_beta"], p["r_tau"]),
            "attention": 1., "last_seen": None, "requests": 0, "answered": 0, "unanswered": 0,
            "intimacy": 0., "self_depth": 0., "partner_depth": 0., "responsive": None,
            "dyad": {"model": RecursiveIdentifier().state, "last_partner_affect": None, "last_expressed": None,
                     "expressed_source": None, "updates": 0},
            "coordination": CoordinationCounts().to_dict(), "lsm": {"n": 0, "sum": 0.}, "last_partner_text": None}


def ensure_bond(state, actor):
    return state["bonds"].setdefault(actor, bond_initial(state["config"]))


def presence(bond, now, tau):
    if bond["last_seen"] is None:
        return 0.
    return math.exp(-max(0., now - bond["last_seen"]) / tau)


def security(state, actor):
    bond = state["bonds"].get(actor)
    if bond is None:
        return 0.
    familiarity = state["relationships"].get(actor, {}).get("familiarity", 0.)
    return bond["responsiveness"] * familiarity


def threat_buffer(state):
    """Fraction of raw threat removed by the most supportive present partner."""
    cfg = state["config"]
    best = 0.
    for actor, bond in state["bonds"].items():
        best = max(best, cfg["buffer_gain"] * security(state, actor) * presence(bond, state["time"], cfg["presence_tau"]))
    return clip(best, 0., 1.)


def focal_partner(state):
    """The configured partner when one is set, else the most recently seen actor."""
    cfg = state["config"]
    if cfg.get("partner"):
        return cfg["partner"]
    seen = [(b["last_seen"], a) for a, b in state["bonds"].items() if b["last_seen"] is not None]
    return max(seen)[1] if seen else None


def observe(state, event, appraisal, affect):
    """Update bond and attachment state after a new occurrence has been appraised."""
    cfg = state["config"]; att = state["attachment"]; now = state["time"]
    actor = event.actor; d = event.data
    bond = ensure_bond(state, actor)
    bond["last_seen"] = now
    support = float(d.get("support", 0.))
    # Comfort (Eq. 4) uses the profile adapted to this partner. Arousal (Eq. 3) uses the
    # stimulation of this occurrence: the goal-relevant prediction-error magnitude (the
    # analogue of the paper's recall error; appraisal surprise is |error|/2) plus the
    # buffered threat it produced. Goal-irrelevant, unthreatening chat lets arousal settle.
    profile = bond["profile"]
    att["comfort"] = comfort_step(att["comfort"], support, 0., profile["tau_h"], profile["beta_h"])
    stimulation = clip(appraisal["goal_relevance"] * 2. * appraisal["surprise"] + affect.get("threat", 0.), 0., 1.)
    att["arousal"] = arousal_step(att["arousal"], stimulation, att["comfort"], cfg["profile"]["tau_ar"], cfg["profile"]["alpha_ar"])
    # A pending request to this partner is answered by support, or counted unanswered.
    if att["pending"] == actor:
        answered = support > .1
        bond["responsiveness"] = responsiveness_step(bond["responsiveness"], att["comfort"] if answered else 0., cfg["profile"]["alpha_resp"])
        bond["profile"] = profile_from_responsiveness(bond["responsiveness"], cfg["profile"]["beta_h"], cfg["profile"]["tau_h"], cfg["profile"]["r_beta"], cfg["profile"]["r_tau"])
        bond["answered" if answered else "unanswered"] += 1
        att["answered"] += answered
        att["pending"] = None; att["pending_since"] = None
    # Dyad identification: partner affect sequence with the agent's expression in between.
    partner_affect = d.get("partner_affect", d.get("approval"))
    if partner_affect is not None:
        dy = bond["dyad"]
        if dy["last_partner_affect"] is not None:
            expressed = dy["last_expressed"] if dy["last_expressed"] is not None else state["mood"]["valence"]
            dy["expressed_source"] = "express" if dy["last_expressed"] is not None else "mood valence (no explicit expression)"
            RecursiveIdentifier(dy["model"]).update(dy["last_partner_affect"], expressed, float(partner_affect))
            dy["updates"] += 1
        dy["last_partner_affect"] = float(partner_affect); dy["last_expressed"] = None
    # Intimacy process (Laurenceau et al. 1998): partner disclosure and perceived responsiveness.
    disclosure = d.get("disclosure")
    if isinstance(disclosure, dict):
        bond["partner_depth"] = .7 * bond["partner_depth"] + .3 * float(disclosure["depth"])
    if "responsiveness" in d:
        bond["responsive"] = float(d["responsiveness"]) if bond["responsive"] is None else .7 * bond["responsive"] + .3 * float(d["responsiveness"])
    if isinstance(disclosure, dict) or "responsiveness" in d:
        _update_intimacy(bond, cfg)
    if isinstance(d.get("text"), str):
        bond["last_partner_text"] = d["text"][:cfg["text_cap"]]


def _update_intimacy(bond, cfg):
    w = cfg["intimacy_weights"]
    responsive = bond["responsive"] if bond["responsive"] is not None else bond["responsiveness"]
    target = (w["self"] * bond["self_depth"] + w["partner"] * bond["partner_depth"] + w["responsiveness"] * responsive) / (w["self"] + w["partner"] + w["responsiveness"])
    bond["intimacy"] = clip(bond["intimacy"] + cfg["intimacy_rate"] * (target - bond["intimacy"]), 0., 1.)


def express(state, partner, valence, text=None, disclosure=None):
    """Record the agent's own expression toward a partner; returns the journaled payload."""
    cfg = state["config"]
    bond = ensure_bond(state, partner)
    bond["dyad"]["last_expressed"] = float(valence)
    item = {"partner": partner, "valence": float(valence), "time": state["time"]}
    if isinstance(disclosure, dict):
        depth = float(disclosure["depth"]) * (1. if disclosure.get("emotional", True) else 0.)
        bond["self_depth"] = .7 * bond["self_depth"] + .3 * depth
        _update_intimacy(bond, cfg)
        item["disclosure"] = {"depth": float(disclosure["depth"]), "emotional": bool(disclosure.get("emotional", True))}
    if isinstance(text, str):
        item["text_chars"] = len(text)
        if bond["last_partner_text"]:
            counts = CoordinationCounts.from_dict(bond["coordination"]).add(bond["last_partner_text"], text)
            bond["coordination"] = counts.to_dict()
            value = lsm_from_proportions(proportions(bond["last_partner_text"]), proportions(text))
            bond["lsm"]["n"] += 1; bond["lsm"]["sum"] += value
            item["lsm"] = value
    return item


def request(state, partner):
    """Bounded help-seeking: consumes the partner's attention budget and opens a pending request."""
    cfg = state["config"]; att = state["attachment"]
    bond = ensure_bond(state, partner)
    if att["pending"] is not None:
        raise ValueError("a support request is already pending")
    if bond["attention"] + 1e-12 < cfg["attention_cost"]:
        raise ValueError("partner attention budget exhausted; wait or let time pass")
    bond["attention"] = max(0., bond["attention"] - cfg["attention_cost"])
    bond["requests"] += 1; att["requests"] += 1
    att["pending"] = partner; att["pending_since"] = state["time"]


def advance(state, dt):
    cfg = state["config"]; att = state["attachment"]
    decay = math.exp(-dt / cfg["bond_tau"])
    att["arousal"] *= decay; att["comfort"] *= decay
    for bond in state["bonds"].values():
        bond["attention"] = min(1., bond["attention"] + dt * cfg["attention_recovery"])
    if att["pending"] is not None and state["time"] - att["pending_since"] > cfg["request_window"]:
        bond = state["bonds"][att["pending"]]
        bond["responsiveness"] = responsiveness_step(bond["responsiveness"], 0., cfg["profile"]["alpha_resp"])
        bond["profile"] = profile_from_responsiveness(bond["responsiveness"], cfg["profile"]["beta_h"], cfg["profile"]["tau_h"], cfg["profile"]["r_beta"], cfg["profile"]["r_tau"])
        bond["unanswered"] += 1
        att["pending"] = None; att["pending_since"] = None


def controls(state):
    """Bond section of the actuation context, for the focal partner."""
    cfg = state["config"]; att = state["attachment"]
    partner = focal_partner(state)
    mode = behavior(att["arousal"], cfg["profile"]["high"], cfg["profile"]["low"])
    out = {"partner": partner, "arousal": att["arousal"], "comfort": att["comfort"], "mode": mode,
           "pending_request": att["pending"], "threat_buffer": threat_buffer(state),
           "exploration_multiplier": {"request": .6, "learn": 1., "explore": 1.15}[mode],
           "firewall": "bond state never modifies reliability, authority, uncertainty or tool permissions"}
    if partner is None or partner not in state["bonds"]:
        out.update(request_support=False)
        return out
    bond = state["bonds"][partner]
    out.update({
        "request_support": mode == "request" and att["pending"] is None and bond["attention"] + 1e-12 >= cfg["attention_cost"],
        "security": security(state, partner), "presence": presence(bond, state["time"], cfg["presence_tau"]),
        "responsiveness": bond["responsiveness"], "profile": dict(bond["profile"]), "attention": bond["attention"],
        "intimacy": bond["intimacy"],
        "reciprocity": {"partner_depth": bond["partner_depth"], "self_depth": bond["self_depth"],
                        "debt": bond["partner_depth"] - bond["self_depth"],
                        "next_depth": clip(bond["partner_depth"] + cfg["disclosure_escalation"], 0., 1.) if bond["partner_depth"] > bond["self_depth"] else bond["self_depth"]},
    })
    dy = bond["dyad"]
    ident = RecursiveIdentifier(deepcopy(dy["model"]))
    p_now = dy["last_partner_affect"] if dy["last_partner_affect"] is not None else 0.
    steering = steer(ident, p_now, state["mood"]["valence"], target=cfg["target_partner_affect"], cap=cfg["coordination_cap"],
                     repair_threshold=cfg["repair_threshold"], repair_amplitude=cfg["repair_amplitude"], min_samples=cfg["steer_min_samples"])
    out["dyad"] = {**steering, "partner_model": ident.params(), "updates": dy["updates"], "last_partner_affect": dy["last_partner_affect"]}
    summary = CoordinationCounts.from_dict(bond["coordination"]).summary()
    out["coordination"] = {**accommodation_assessment(summary["aggregated_3"]), "exchanges": summary["exchanges"],
                           "lsm_mean": (bond["lsm"]["sum"] / bond["lsm"]["n"]) if bond["lsm"]["n"] else None,
                           "cap": cfg["coordination_cap"]}
    return out


def validate(state):
    from ..schema import finite
    att = state["attachment"]
    finite(att["arousal"], 0, 2); finite(att["comfort"], 0, 2)
    for bond in state["bonds"].values():
        finite(bond["responsiveness"], 0, 1); finite(bond["attention"], 0, 1); finite(bond["intimacy"], 0, 1)
        finite(bond["self_depth"], 0, 1); finite(bond["partner_depth"], 0, 1)
        finite(bond["profile"]["beta_h"], 0, 1); finite(bond["profile"]["tau_h"], 0)
        for v in bond["dyad"]["model"]["theta"]:
            finite(v)
