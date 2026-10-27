"""Attachment-style arousal regulation with an adaptive help-seeking profile.

Equations (3)-(7) and the parameter tables of Hiolle, Lewis & Cañamero (2014),
*Arousal regulation and affective adaptation to human responsiveness by a robot
that explores and learns a novel environment*, Frontiers in Neurorobotics 8:17
(CC BY), are implemented as pure functions. In the original, stimulation came
from recall error and map-weight change in two neural networks fed by camera
images; callers supply any stimulation signal in [0, 1]-like range.

Published quirks are preserved and labelled rather than silently corrected:
the responsiveness decrease in Eq. (5) uses (1 - Resp), not Resp. Arousal is
floored at zero here (the paper does not state a floor; see `floor` argument).

Usage hazards found by the independent reproduction (docs/reproductions/attachment_robot.md):

* Eq. (4) tests ``contact > 0 or face > 0``. If callers pass Eq. (1) traces, which
  decay geometrically and never reach zero, the ``beta_h`` branch becomes
  unreachable after the first contact and the two profiles behave alike. Pass
  raw or truncated sensor values (the platform passes per-event support).
* Eqs. (6)-(7) give ``beta_h = 1`` at ``Resp = 0``, so the comfort trace then
  stops decaying between interventions; the platform additionally decays comfort
  with elapsed time.
* Evaluating Eq. (5) only on steps where arousal is in the request band cannot
  yield a net increase under a caregiver whose response arrives after a latency,
  because comfort removes arousal from that band immediately. The platform
  therefore updates responsiveness once per resolved request instead.
"""
from __future__ import annotations

# Table 1 (fixed profiles) and Table 2 (adaptive regulation) of Hiolle et al. (2014).
PROFILES = {
    "needy":       {"alpha_ar": .6, "tau_ar": 5, "tau_h": 3, "beta_h": .70, "high": .6, "low": .4},
    "independent": {"alpha_ar": .6, "tau_ar": 5, "tau_h": 8, "beta_h": .95, "high": .6, "low": .4},
}
ADAPTIVE = {"alpha_ar": .8, "tau_ar": 3, "tau_h": 3, "beta_h": .7, "alpha_resp": .03,
            "resp0": .5, "r_tau": 7, "r_beta": .3, "high": .6, "low": .4}
COMFORT_FLOOR = .1   # Comf(t) <= 0.1 counts as "no comfort" in Eq. (3) and Eq. (5)


def perception_trace(previous, sensor, alpha_sensor):
    """Eq. (1): binary sensors leave an exponentially decaying trace."""
    return 1. if sensor else alpha_sensor * previous


def comfort_step(comfort, contact, face, tau_h, beta_h):
    """Eq. (4): comfort window average while a human intervenes, otherwise a trace decay."""
    if contact > 0 or face > 0:
        return (comfort * tau_h + contact + face) / (tau_h + 1)
    return beta_h * comfort


def arousal_step(arousal, stimulation, comfort, tau_ar, alpha_ar, floor=0.):
    """Eq. (3): exponential average of stimulation without comfort; comfort subtracts directly."""
    if comfort <= COMFORT_FLOOR:
        value = (tau_ar * arousal + stimulation) / (tau_ar + 1)
    else:
        value = arousal - alpha_ar * comfort
    return value if floor is None else max(floor, value)


def behavior(arousal, high=.6, low=.4):
    """Winner-take-all selection of Figure 4: request help, stay and learn, or explore."""
    if arousal >= high:
        return "request"
    if arousal > low:
        return "learn"
    return "explore"


def responsiveness_step(responsiveness, comfort, alpha_resp):
    """Eq. (5), evaluated while the help-seeking behaviour is active.

    The published decrease term is alpha*(1-Resp), reproduced verbatim.
    """
    if comfort > COMFORT_FLOOR:
        value = responsiveness + alpha_resp * (1 - responsiveness)
    else:
        value = responsiveness - alpha_resp * (1 - responsiveness)
    return min(1., max(0., value))


def profile_from_responsiveness(responsiveness, beta0=.7, tau0=3, r_beta=.3, r_tau=7):
    """Eqs. (6)-(7): a responsive partner moves the profile toward 'needy' (short comfort memory)."""
    return {"beta_h": beta0 + (1 - responsiveness) * r_beta, "tau_h": tau0 + (1 - responsiveness) * r_tau}
