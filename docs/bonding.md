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

### Gottman, Swanson & Murray (1999) and Gottman et al. (2003): dyadic affect model

Computational reproduction of the model's stated properties and of its published
estimation procedure from the published group means; no couple data
([report](reproductions/marital_dynamics.md)).

| Published statement | Outcome |
|---|---|
| Sigmoidal influence gives one, three or five steady states | Step ojive: at most three fixed points, all stable (7,056 couples); counting the jump crossings of the completed null clines gives exactly 1/3/5 in every couple. Smooth sigmoid (1,488 couples): counts always 1, 3 or 5, with stable minus unstable equal to one. Reproduced once the reading is made explicit |
| Uninfluenced steady state a/(1-r) | Reproduced exactly |
| Published influenced steady-state means | Not matched within 0.5 for stable, happy-stable or unhappy-stable groups (distance 1.16-1.85) for any plateau pair; expected, since they are means of per-couple estimates |
| Divorced parameters attract negatively; initial conditions decide the end point | Reproduced qualitatively (divorced: 100% negative basins at plateaus up to 8/5); period-2 threshold-straddling cycles capture 10-22% of initial conditions at large plateaus, which the papers do not mention |
| The published estimation algorithm | Exact on noiseless data with partner zeros; at 150 points with SD-3 process noise it is near-unbiased but imprecise (inertia RMSE 0.24-0.29; positive threshold biased -3.2 to -5.1). Widening the zero subset to within 1 halves the inertia error; the joint and online bilinear estimators have inertia RMSE 0.08-0.10 but a misspecification bias for off-zero groups |
| Damping may create a positive-positive attractor where none existed (2003) | Not reproduced in 0/360 configurations; with constant-trigger damping and monotone influence it is provably impossible (the damped map is dominated by the undamped one). The authors' own damping algorithm was not available; reported as untested, not refuted |

Consequence for the platform: the agent's online identifier should be read with
the E3 precision in mind (inertia to about 0.1 after 150 exchanges), the
steering recommendation is therefore capped and labelled by `model_trusted`, and
the closed-loop fixed point it reports is a model prediction, not an observed
attractor.

### Blevins (2026) and French et al. (2024): do language models over-converge?

Scoped replication with the pinned Qwen2.5-Coder-1.5B-Instruct model on 200
locked DailyDialog contexts, 400 recorded calls, zero provider errors
([report](reproductions/llm_convergence.md)). Eight-marker convergence toward the
target utterance (Eq. 1, aggregated 3), paired item bootstrap:

| Response set | Convergence [95% CI] | Shuffled-pair baseline | Content-word proxy | Mean tokens |
|---|---:|---:|---:|---:|
| Human next turn | +.034 [+.007, +.061] | +.001 | .062 | 11.9 |
| Model, plain prompt | +.025 [+.002, +.050] | +.001 | .081 | 8.4 |
| Model, bounded-coordination instruction | +.032 [+.009, +.056] | +.000 | .081 | 8.5 |

| Published statement | Measured | Status |
|---|---|---|
| LLMs converge about twice as much as users on function words (+.033 LIWC mean; +.021 on these eight markers) | plain minus human **-.009 [-.041, +.025]** | **Not reproduced**; a gap of +.025 or more is excluded on this lexicon, the published eight-marker gap is barely inside the interval |
| Noun convergence .442 vs .141 | content proxy .081 vs .062, +.019 [-.017, +.063] | Same sign, about a tenth of the size, interval includes zero; proxy is not noun-lemma convergence |
| Human-human DailyDialog convergence .036 | +.034 (eight markers), +.041 (nine) | Means agree; per-marker pattern differs |
| Platform mechanism: bounded-coordination instruction reduces over-accommodation | bounded minus plain **+.007 [-.009, +.023]**; replies not emptied | Null: there was no excess to remove with this model |

The platform's coordination control therefore has no demonstrated effect on
this model, which did not show the published excess; all three sets fall
`within` the published human band by `accommodation_assessment`. A shuffled
baseline bug in the first analysis (permuting pairs together) was corrected
after the run and is recorded in the manifest; no estimate changed.

## Platform-level simulation checks

Protocol, runner and raw runs: `configs/bonding_platform.json`,
`regesh/bonding/experiment.py`, `results/bonding/platform/` (first protocol) and
`configs/bonding_platform_v2.json`, `results/bonding/platform_v2/` (second,
post hoc protocol). Both use the full agent with in-memory storage, 40 seeds,
common random numbers between a bonded agent and a reference agent without bond
state, and bootstrap intervals over seeds. Partners are simulated by declared
equations; no human took part.

### First protocol (pre-registered)

**Attachment loop** (240 turns, episodic low-control setbacks, partner answers a
request with probability q). Bonded minus reference, paired by seed:

| q | Cumulative affect threat | Mean mood valence | Requests (bonded) | Answered fraction | Final responsiveness | Final tau_h | Mean threat buffer |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | -10.58 [-10.71, -10.46] of 71.9 | +.024 | 3.35 | .99 | .547 | 6.17 | .246 |
| 0.5 | -10.40 [-10.53, -10.29] | +.024 | 3.23 | .46 | .495 | 6.53 | .242 |
| 0.0 | -10.34 [-10.46, -10.23] | +.024 | 3.08 | .00 | .452 | 6.83 | .240 |

The buffer removes about 15% of cumulative threat, but almost independently of
partner behaviour: at the published per-resolution rate (alpha .03) and about
three resolved requests per run, measured responsiveness moves only from .50 to
.55 or .45, so security is dominated by its prior and by familiarity. The
direction of the responsiveness and profile adaptation is correct and
monotone in q; its size is not useful at this interaction density. This is a
measured limitation of the first design and the reason for the second protocol.

**Dyadic steering** (120 turns, published-parameter partners scaled to [-1, 1],
declared influence slopes, process noise .05; final-window partner affect, mean
over the last 40 turns; contrasts versus expressing own mood, paired by seed):

| Partner | Own mood | Mirror (unbounded) | Steer, cap .5 | Steer, cap .2 | Steer minus own mood |
|---|---:|---:|---:|---:|---:|
| Stable husband (r .40, u +.03) | +.029 | +.048 | +.286 | +.116 | +.257 [.256, .257] |
| Divorced husband (r .30, u -.08) | -.082 | **-.281** | +.136 | -.008 | +.218 [.218, .218] |
| Step-ojive mismatch (u -.05) | -.051 | -.051 | +.282 | -.051 | +.333 [.333, .333] |

Capped steering raised the simulated partner's affect for every partner,
including the out-of-model one, while unbounded mirroring lowered the negative
partner's affect by .20 (0% positive turns), the quantitative form of the
argument against maximal accommodation. Two honest caveats: the repair trigger
never fired (no partner fell to -.4), so repair is untested here; and because
all three partners respond positively to positive expression, the model-based
recommendation saturated at the cap and was indistinguishable from fixed
positivity. The second protocol adds a partner for whom positivity backfires.
The online identifier's inertia error was about -.04 under steering and -.20 to
+.13 under mirroring, where the agent's expression is collinear with the
partner's state.

### Second protocol (post hoc, declared in `configs/bonding_platform_v2.json`)

Two changes motivated by the first run: the Beta evidence-count responsiveness
estimator, `(answered + 1) / (answered + unanswered + 2)`, as the bond's
estimator (with a variant that also holds arousal on goal-irrelevant events),
and a reactant partner whose affect falls when the agent is positive, plus a
fixed-positive comparator policy. The first protocol's results are retained.

**Attachment loop**, bonded minus reference by seed, and measured
responsiveness:

| q | Variant | Requests | Final responsiveness | Final tau_h | Mean threat buffer | Cumulative threat, bonded minus reference |
|---:|---|---:|---:|---:|---:|---:|
| 1.0 | beta | 3.1 | **.78** | 4.5 | .282 | **-12.12 [-12.37, -11.88]** |
| 0.5 | beta | 3.2 | .49 | 6.6 | .253 | -10.87 [-11.14, -10.64] |
| 0.0 | beta | 3.1 | **.23** | 8.4 | .240 | -10.34 [-10.46, -10.23] |
| 1.0 | beta, hold arousal on neutral events | 16.7 | .95 | 3.4 | .338 | -14.53 [-14.75, -14.31] |
| 0.5 | beta, hold | **32.1** | .51 | 6.5 | .253 | -10.99 [-11.28, -10.73] |
| 0.0 | beta, hold | 9.0 (then stops at the floor) | .09 | 9.4 | .240 | -10.36 [-10.48, -10.25] |

With the evidence-count estimator, measured responsiveness tracks partner
behaviour within a few resolved requests, the profile moves toward "needy"
under a responsive partner and toward "independent" under an unresponsive one,
and the threat reduction now depends on partner behaviour (-12.1 vs -10.3 of
71.9). The estimator is therefore the platform default. Holding arousal on
neutral events increases help-seeking strongly, and under intermittent
responsiveness (q = .5) it produces 32 requests in 240 turns, the
anxious-resistant pattern of intermittent reinforcement; that option stays off
by default, and the new responsiveness floor stops requests to an unresponsive
partner after nine unanswered ones rather than at the attention budget.

**Dyadic steering**, final partner affect (mean of the last 40 turns):

| Partner | Own mood | Mirror | Fixed positive | Steer (cap .5) | Steer minus own mood | Fixed positive minus own mood |
|---|---:|---:|---:|---:|---:|---:|
| Stable husband | +.029 | +.048 | +.286 | +.286 | +.257 | +.257 |
| Divorced husband | -.082 | -.281 | +.136 | +.136 | +.218 | +.218 |
| Step-ojive mismatch | -.051 | -.051 | +.282 | +.282 | +.333 | +.333 |
| **Reactant mismatch** (positivity lowers affect) | -.052 | -.306 | **-.297** | **-.053** | -.001 [-.002, -.000] | **-.245 [-.245, -.244]** |

Against the three partners who respond to positivity, the identifier-based
recommendation and fixed positivity are indistinguishable (both saturate at the
cap). Against the reactant partner, fixed positivity and mirroring each cost
about .25 of partner affect while the identifier-based recommendation learned to
stay neutral (agent valence .002) and caused no harm. That is the measured value
of the online model: it does not add positivity when positivity works, it
withholds it when positivity backfires. The repair trigger again never fired
and remains untested; a tighter cap of .2 cannot cross the ojive partner's
threshold and has no effect there.

Figures: `results/bonding/figures/platform.png` and `platform_v2.png`. The
record-only audit (`results/bonding/audit.json`) recomputes every summary mean
from the raw runs and verifies all six manifests.

## Learned components: what a small network adds, measured

Two follow-up studies asked whether a small neural network trained on
embeddings from the same local model that generates replies improves on the
hand-specified and linear parts of the bond. Both use the shared NumPy MLP
(`regesh/bonding/mlp.py`) and a content-addressed cache of mean-pooled
1536-dimensional embeddings (`regesh/bonding/embeddings.py`), with frozen
protocols, hash-selected training subsets and single evaluations on official
test splits.

### Learned partner affect model versus the online identifier

Task: predict the partner's next affect from history plus the agent-role turn
on DailyDialog (partner = first speaker; 2,258 training dialogues, full test
split of 1,000 dialogues, 3,040 examples, 81% neutral targets; seven-emotion
valence map declared; [report](bonding_partner_model.md)). Dialogue-level
bootstrap intervals:

| Model | RMSE | Balanced sign accuracy | Next-emotion macro-F1 |
|---|---:|---:|---:|
| Persistence (repeat the partner's last affect) | .266 | .49 | **.545** |
| Platform identifier, diffuse start (previous default) | .275 [.259, .291] | .64 | n/a |
| Same identifier warm-started from the pooled fit | **.244** [.233, .256] | .80 | n/a |
| Ridge on 13 scalar features | **.235** | .79 | .196 |
| MLP, scalar features only | **.235** | **.81** | .128 |
| MLP, embeddings plus scalars (headline) | .242 [.230, .253] | .80 | .128 |
| MLP, embeddings only | .270 | .53 | .128 |

Findings. Embeddings added nothing to aggregate next-affect accuracy and cost a
small, significant amount (headline minus scalar-only MLP: RMSE +.007 [+.004,
+.009]); validation selected the strongest penalties, nearly switching them
off. The learned model's advantage over the platform identifier (RMSE -.033
[-.044, -.023]) is a cold-start effect: warm-starting the same four linear
parameters from the pooled training fit recovers it (-.031 [-.038, -.024]).
No learned model predicted any rare emotion class; persistence did (anger,
disgust and fear F1 .59-.70), because emotion persists within a speaker.
Candidate ranking with embeddings preferred the actual next agent turn over a
random one in 71% of positive-outcome items but was at chance for negative
outcomes, and the comparison confounds affect with topical coherence. Protocol
amendments (projection, whitening, a validation-selected penalty grid after
the frozen configuration memorised) were made before any test label was read
and are dated in the configuration.

Consequence adopted in the platform: the partner identifier now warm-starts
from the pooled DailyDialog prior with unit covariance (`identifier_prior`
"dailydialog-pooled-v1"; the diffuse start remains selectable), and persistence
is the recommended predictor of the next emotion class. The pooled fit's
positive slope exceeds its negative one, the reverse of the asymmetry Gottman
reports for marital conflict; it is a property of this scripted everyday
corpus, not a general law. The learned model is retained as an evaluated
alternative, not deployed.

LEARNED_SENSOR_SECTION

## Claim ledger

| Label | Scope |
|---|---|
| Established precedent | Arousal regulation by a caregiver, adaptive attachment profiles, coupled dyadic affect equations with attractors and repair, mid-range contingency, linguistic coordination and style matching, the intimacy process model, relational agents and chatbot attachment. None is claimed as new. |
| Engineering synthesis | One replayable, hash-journaled bond state per partner that combines these mechanisms inside the cognitive-control loop, with shared pure-function modules used by both the platform and the reproductions, an attention budget, a responsiveness floor and a trust firewall. |
| Measured result | Default agents unchanged (demo digest identical). Reproductions: Hiolle profile contrasts and the complex-environment traversal ratio reproduce, absolute pace and the literal adaptive rule do not; the step ojive admits at most three fixed points, the published counts are null-cline crossings or smooth-form roots, the published estimator is imprecise at 150 points, damping never created a positive attractor; two of four Supreme Court coordination asymmetries reproduce; LLM over-convergence does not reproduce with the 1.5B model and the bounded instruction is a null. Platform: a paired cumulative threat reduction of 10-15% that depends on partner responsiveness only under the evidence-count estimator; identifier-based steering equals fixed positivity for cooperative partners and avoids its harm for a reactant one; repair untested. |
| Measured result (learned components) | On DailyDialog, embeddings from the generating model add nothing to next-affect prediction beyond scalar affect features; the identifier's deficit was cold start and is fixed by a pooled warm start, now the default; persistence predicts the next emotion class best. SENSOR_LEDGER |
| Speculative hypothesis | That these mechanisms improve collaboration, trust calibration or wellbeing with real users. No human study was run; the simulated partners follow declared equations; benefit to users is unestablished. |

What would change the ledger: a matched study with real users and an ordinary
non-affective adaptive controller as comparator, a model that does show the
published over-convergence (to test the coordination control), partners whose
affect falls below the repair threshold (to test repair), and a longitudinal
measurement of help-seeking rates against user-reported burden.

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
