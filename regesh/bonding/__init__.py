"""Pair-bonding mechanisms: attachment regulation, dyadic coupling and synchrony.

These modules implement published equations (Hiolle et al. 2014; Gottman,
Swanson & Murray 1999; Gottman et al. 2003; Danescu-Niculescu-Mizil et al.
2011/2012; Ireland et al. 2011; Laurenceau et al. 1998) as transparent,
replayable engineering. A bond is functional state that regulates help-seeking,
threat buffering, expressed affect and accommodation. It never alters factual
reliability, authority or tool permissions, and establishes no subjective
experience or biological fidelity.
"""
from .synchrony import CATEGORIES, exhibits, lsm, coordination, CoordinationCounts
from .attachment import arousal_step, comfort_step, responsiveness_step, profile_from_responsiveness, behavior
from .dyad import ojive, bilinear, step as dyad_step, simulate as dyad_simulate, steady_states, RecursiveIdentifier

__all__ = ["CATEGORIES", "exhibits", "lsm", "coordination", "CoordinationCounts",
           "arousal_step", "comfort_step", "responsiveness_step", "profile_from_responsiveness", "behavior",
           "ojive", "bilinear", "dyad_step", "dyad_simulate", "steady_states", "RecursiveIdentifier"]
