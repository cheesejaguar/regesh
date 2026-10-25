# Pair bonding with a user: mechanisms, reproductions and measured limits

Date: 2026-10-25. This report describes three additions to the affect platform
that let an agent form and regulate a functional **bond** with its user, the
published models they are built on, four reproductions of published work, a
platform-level simulation, and the safeguards. Measured numbers are in the
Results sections; everything else is design. Nothing here establishes that the
mechanisms benefit real users, that the agent experiences attachment, or that
the models are biologically faithful.

## Why bonding, and why these three mechanisms

Attachment research describes a bond operationally: proximity seeking, safe
haven, secure base and separation distress, formed by a history of contingent,
responsive care (Bowlby; Ainsworth et al. 1978). Three measurable processes
carry most of the empirical weight and each has a computational model with
published equations or measures:

1. **Arousal regulation by a responsive partner** (Hiolle et al. 2012, 2014):
   a robot's arousal rises with novelty, is lowered by a caregiver's comfort, and
   a "needy"/"independent" profile adapts to the caregiver's measured
   responsiveness. The same regulation yields a secure base: a present,
   responsive partner reduces the threat response (Coan et al. 2006).
2. **Coupled dyadic affect dynamics** (Cook et al. 1995; Gottman et al. 1999,
   2003): each partner's affect follows inertia, an uninfluenced set point and a
   nonlinear influence of the other; the dyad has attractors; repair at a
   negativity threshold predicts stability. Mid-range, not maximal,
   interactive contingency predicts secure attachment (Jaffe et al. 2001).
3. **Synchrony and reciprocity** (Niederhoffer & Pennebaker 2002; Ireland et al.
   2011; Danescu-Niculescu-Mizil et al. 2011, 2012; Laurenceau et al. 1998;
   Aron et al. 1997): function-word coordination tracks rapport and power;
   reciprocal, escalating disclosure with perceived responsiveness produces
   intimacy; LLMs already over-converge to users (Blevins 2026).

Each change implements the published equations as pure functions in
`regesh/bonding/` and reuses them both in the platform and in the
reproductions, so the evaluated mechanism is the deployed one.

## Change 1: attachment regulation with budgeted help-seeking and a secure base

State: agent-level `attachment` (arousal, comfort trace, pending request) and
per-partner responsiveness, adapted comfort profile and attention budget.
Equations (3)-(7) of Hiolle et al. (2014) with their Table 2 parameters:
stimulation is the goal-relevant prediction-error magnitude plus the threat
the occurrence produced; comfort comes from `support` evidence; a request is
answered by the partner's next occurrence carrying support, or counted as
unanswered; responsiveness updates once per resolved request; the profile moves
toward "needy" (short comfort memory) under a responsive partner and toward
"independent" otherwise. Threat of any new occurrence is multiplied by
`1 - gain * responsiveness * familiarity * presence`.

Controls: `mode` (request / learn / explore by arousal band), `request_support`
(only when no request is pending and the partner's attention budget covers the
cost), `security`, `threat_buffer`, and an exploration multiplier by band.

## Change 2: dyadic affect model with online identification and bounded steering

The partner's equation `P_{t+1} = b + r P_t + s_neg min(A_t,0) + s_pos max(A_t,0)`
(the 2003 bilinear form) is identified online by recursive least squares from
the sequence of partner affect (`partner_affect` evidence, falling back to
`approval`) and the agent's expressed affect (`express()`, or its mood valence if
it did not express). The control recommends an expressed valence that moves the
partner one step toward a positive target, capped at `coordination_cap` around
the agent's own valence, adds a repair jump when the partner is at or below the
negativity threshold, and reports the predicted closed-loop fixed point. The cap
exists because full mirroring is the "heightened contingency" that predicts
insecurity, and because it would amplify a negative partner.

## Change 3: synchrony and reciprocity channel

`express(partner, text=...)` pairs the agent's utterance with the partner's last
utterance and accumulates the eight-marker coordination counts of Eq. (1) of
Danescu-Niculescu-Mizil et al. (2012) and Ireland et al.'s language style
matching, using an authored open function-word lexicon (not LIWC). The control
reports mean convergence, its excess over the published human-human band
(centre .036, half-width .03; the LLM in the same corpus reached .068) and a
band label, so a host can damp over-accommodation. Disclosure evidence and
perceived responsiveness update intimacy with Laurenceau et al.'s standardized
path weights (self .22, partner .32, responsiveness .36; emotional disclosure
only), and the control recommends the next disclosure depth as the partner's
depth plus a bounded step only while a reciprocity debt exists.

## Safeguards

The bond never changes reliability, authority, epistemic uncertainty or
permissions (tested). Requests are budgeted and time out; no distress or guilt
is simulated at disengagement, excluding the "caregiving-system capture"
tactics documented by De Freitas (2026). All bond variables are inspectable,
replayable and interveneable. See `bonding_disclosure.md`.

## Reproductions of published work

All four are independent implementations or scoped replications; none is a
rerun of the authors' code. Full protocols, declared ambiguities, raw outputs and
hashes are in the linked reports and `results/papers/`.

### Hiolle, Lewis & Cañamero (2014): arousal regulation and adaptation

Independent simulation with the shared equations, Table 1/2 parameters and a
declared stimulation substitute (2,100 runs, 601,324 steps;
[report](reproductions/attachment_robot.md)). Needy-minus-independent contrasts
with a responsive caregiver, 100 seeds, paired bootstrap:

| Published observation | Measured | Status |
|---|---|---|
| Needy requests comfort much more often | +2.00 requests (simple), +3.00 (complex); intervals exclude zero | Direction reproduced |
| Needy spends more time at medium arousal and learns more; independent explores more | learn-band fraction +.19 / +.15; explore fraction -.24 / -.22; learning steps +15 / +16 | Direction reproduced |
| Independent comfort trace lasts up to 10 s | 12.3 s above the .1 floor (needy 3.0 s) | Consistent; arithmetic of beta_h = .95 |
| Complex environment: 140 s vs 170 s for two traversals (ratio 1.21) | 26.0 s vs 31.4 s, ratio 1.207 | Ratio reproduced; **absolute seconds do not** |
| Simple environment: similar pace for both profiles | 26.7 s vs 31.6 s, ratio 1.186 | **Not reproduced in substance** |
| Adaptive robot: responsiveness rises under a responsive human (5-140 s), falls after 150 s | Literal Eq. 5: falls from .5 to .014 by 100 s, beta_h/tau_h to the independent end; rise-then-fall appears only when the regulatory behaviour is assumed to persist through a 3 s comfort episode (.977 at 150 s, .916 at end) | **Primary direction not reproduced**; reproduced under two unspecified semantics, reported as sensitivities |

Consequence for the platform: the per-step Eq. 5 cannot raise responsiveness when
comfort arrives after a latency, so the bond updates responsiveness once per
resolved request at the published rate. The runs are nearly deterministic across
seeds, so the intervals describe declared noise only.

### Danescu-Niculescu-Mizil, Lee, Pang & Kleinberg (2012): coordination and power

Independent implementation of the Supreme Court analysis on the public ConvoKit
corpus with the shared open-lexicon measure: 204/204 argument sessions, 49,281
Justice-lawyer exchanges (paper 50,389), 11 Justices, 312 lawyers
([report](reproductions/coordination_power.md)). Percent scale, Welch t across
(speaker, case) entities, ConvoKit 7/7 thresholds fixed before any result:

| Published claim | Aggregated 3 difference (points) | p | Status |
|---|---:|---:|---|
| Lawyers coordinate more toward Justices than conversely (Fig. 4) | +1.18 [0.77, 1.57] | <.0001 | Reproduced; 6/8 markers in direction, 4 significant |
| Lawyers coordinate more toward unfavorable than favorable Justices (Fig. 5a) | +0.98 [0.37, 1.60] | .002 | Reproduced; 8/8 markers in direction |
| Lawyers coordinate more toward the Chief than Associate Justices (p<.01) | -0.35 | .50 | Not reproduced; direction only under the loosest threshold |
| Favorable Justices coordinate more toward lawyers (Fig. 5b) | +0.07 | .83 | Not reproduced; adverbs significant in the opposite direction |

Absolute levels are not comparable with the paper because the lexicon is not
LIWC. The measure recovers the two large published asymmetries and not the two
smaller ones; this run cannot separate lexicon, threshold and data-processing
explanations for the failures.

RESULTS_REPRODUCTIONS_REMAINING

## Platform-level simulation checks

RESULTS_PLATFORM

## Claim ledger

CLAIM_LEDGER

## Reproduce

```bash
python -m regesh.reproductions.attachment_robot --output results-new/attachment-robot
python -m regesh.reproductions.marital_dynamics --output results-new/marital-dynamics
python -m regesh.reproductions.coordination_power --cache .cache/convokit/supreme --output results-new/coordination
python -m regesh.reproductions.llm_convergence --base-url http://127.0.0.1:8080/v1 --model regesh-qwen-coder --output results-new/llm-convergence
OPENBLAS_NUM_THREADS=1 python -m regesh.bonding.experiment --config configs/bonding_platform.json --output results-new/bonding-platform
python scripts/audit_bonding.py --output results-new/bonding-audit.json
python scripts/plot_bonding.py --output results-new/figures
python -m unittest discover -s tests -v
```
