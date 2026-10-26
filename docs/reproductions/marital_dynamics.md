# Gottman, Swanson and Murray (1999) and Gottman et al. (2003): dyadic affect model

Execution date: **2026-10-25 UTC**. Primary sources: [The mathematics of marital
conflict: Dynamic mathematical nonlinear modeling of newlywed marital interaction](https://doi.org/10.1037/0893-3200.13.1.3),
*Journal of Family Psychology* 13(1):3-19, and the repair/damping extension in
[Observing gay, lesbian and heterosexual couples' relationships: Mathematical modeling of conflict interaction](https://doi.org/10.1300/J082v45n01_04),
*Journal of Homosexuality* 45(1):65-91. Both are peer-reviewed empirical papers.
This study is a **computational reproduction of the model's stated properties and
of its estimation procedure from published group-mean parameters**. It uses no
couple data, so it cannot reproduce the papers' divorce-prediction statistics,
and it does not test Regesh's controllers.

## Scope and source verification

The extracted texts of both papers were read in full (SHA-256
`4a222a00b47cf6a3cf6f5f927266cb43daf23ce8df59bf1c07f3030ccefd31ae` for 1999 and
`911685e8cdfb988d5284f572eb9ef6aa5016fa754831d23d454d9af58e58b9cc` for 2003; the
texts are not redistributed here). The 1999 model (its Equations 1-2), the ojive
influence form (Figure 1A), the steady-state discussion, the "Computational
Algorithms" section and Tables 1-2 were transcribed from prose and tables, not
from figure pixels. The 2003 paper supplies the bilinear form, the verbal
definition of repair and damping triggers, and group means for repair
effectiveness (2.78-3.99), repair thresholds (-4.64, -6.04) and damping
effectiveness (-3.11 to -5.03). Neither paper publishes the ojive plateau values,
the bilinear slopes, the damping threshold, or per-couple parameters. The authors'
estimation program ("can be supplied on request") was not obtained or executed.

The model is **imported from `regesh/bonding/dyad.py`** (`ojive`, `sigmoid_ojive`,
`bilinear`, `repair`, `damping`, `step`, `simulate`, `steady_states`,
`fixed_points_numeric`, `fit_published_algorithm`, `fit_joint_bilinear`,
`RecursiveIdentifier`, `PUBLISHED_1999`) and is not re-implemented. The runner
adds only grid construction, the null-cline crossing count described below,
the noise model, and bookkeeping. A copy of the module as executed is saved
beside the outputs as `dyad_source.py`.

**Fidelity label: computational reproduction of model properties and of the
published estimation procedure; no original data.** Nothing here estimates
feelings, and no claim about real couples follows from these simulations.

## Fixed protocol

The [configuration](../../configs/reproductions/marital_dynamics.json) was
frozen at 23:43:23 UTC, before the full run started at 23:49:47 UTC. No grid,
noise level or threshold was changed after seeing results. A two-minute pilot
timed the solvers and compared Newton seed densities on 150 random couples;
its only influence on the protocol was the size of the numeric grid and the
seeding strategy, both declared before the run. Published Table 1/2 means are
copied verbatim from `PUBLISHED_1999`; the test suite asserts the copy matches.

Parameter construction follows the paper: `a = u (1 - r)` from the published
inertia `r` and uninfluenced steady state `u = a/(1 - r)`. A spouse's published
thresholds are taken as the thresholds of the partner's response to that
spouse (the husband's negative threshold is "how negative he must become
before his wife responds"), so the husband's thresholds parameterize `I_hw(H)`
in the wife's equation. The alternative mapping was not run. Fifteen further
ambiguities are declared in the configuration, including the two readings of
"O-Jive or sigmoidal", the constant-trigger reading of repair/damping, and the
within-paper inconsistency between the 1999 prose (husbands' influenced steady
states -1.89 and 0.72, which are the uninfluenced means) and Table 1 (-1.49 and
0.71); the table values are used.

| Experiment | Declared design |
|---|---|
| E1 structural | Step ojive: inertia {.3,.4,.5} crossed between spouses, uninfluenced {-3..3} per spouse, shared plateau pair from {1,3,5,8}², Table 1 stable-group thresholds; 7,056 couples. Smooth sigmoid (logistic ramps centred on the thresholds): width 1.0 on shared inertia × uninfluenced {-3,-1.5,0,1.5,3}² × 16 plateau pairs (1,200 couples); widths 0.5 and 2.0 at inertia .4 on {-3,0,3}² (288 couples). |
| E2 published | Four groups; plateau pair swept 0.5-10 in 0.5 steps (400 pairs per group); match if both coordinates within 0.5 of the published influenced means. Basins: 625 initial conditions on the integer-step-2 grid of [-24,24]², 150 iterations without clipping, four plateau pairs. |
| E3 recovery | Four groups, plateau ±3, process noise SD {1,3,5}, 1,000 replications per cell, 150 points, initial scores uniform on {-6..6}, rounded to integers and clipped to [-24,24]. Published algorithm with exact zeros (`zero_tol` 0) and, at SD 3, with `zero_tol` 1; `fit_joint_bilinear` at all SDs; `RecursiveIdentifier` (λ=.98) at SD 3. Integer threshold grid 1..24. Bootstrap 2,000 resamples. |
| E4 repair/damping | Bilinear slopes {(.5,.3),(.3,.2),(.7,.4)}, repair effectiveness {2.78,3.99} × thresholds {-4.64,-6.04}, damping effectiveness {-3.11,-5.03} × thresholds {4,6,8}, applied identically to both spouses; 420 configurations; stable fixed points counted per sign quadrant; cycle check from the uninfluenced point when no fixed point exists. |

Counting steady states needs one definition that the 1999 paper leaves
implicit. With the **step** ojive each null cline is three disjoint
axis-parallel segments, so a strict fixed point is a level-level crossing and
at most three can exist (a vertical segment has one `W`, inside at most one
horizontal segment's range). The paper's graphical "unstable" steady states are
where a *connecting* segment at a jump of one null cline meets a connecting
segment of the other. The runner counts those jump crossings separately
(`n_jump_crossings`) and reports `n_completed = n_fixed + n_jump_crossings`.
Under the **smooth** sigmoid reading these crossings become genuine unstable
fixed points, found by the module's damped Newton solver from the union of a
9×9 grid on [-24,24]² and an 8×8 grid on [-21,21]²; a declared random subsample
of 60 couples was rechecked against the module's dense 25×25 default.

## Execution and outputs

```bash
OPENBLAS_NUM_THREADS=1 python -m regesh.reproductions.marital_dynamics \
  --config configs/reproductions/marital_dynamics.json \
  --output results/papers/marital_dynamics-new
python -m unittest tests.test_marital_dynamics_reproduction -v
```

The output directory must be empty. The run writes the configuration copy,
the runner source, the model-module copy, `summary.json`, `manifest.json` with
SHA-256 of every file, the raw per-couple records
(`e1_structural.jsonl.gz`, `e1_reference_check.jsonl.gz`,
`e2_plateau_sweep.jsonl.gz`, `e3_recovery.jsonl.gz`, `e4_repair_damping.jsonl.gz`)
and NumPy arrays (`e2_basins.npz` with every final state, `e3_series.npz` with
all 12,000 simulated integer series). Every seed derives from indices, so the
three-process pool used here does not affect any number. The run took
**406.9 s wall** on a CPU shared with two other experiments and an inference
server (E1 189.7 s, E2 2.4 s, E3 208.0 s, E4 0.1 s); Python 3.13.17, NumPy 2.5.3.
Eight new unit tests check a hand-calculated step, the `a/(1-r)` steady state,
the three-strict/five-completed symmetric ojive, a constructed five-piece couple
with five strict fixed points, exact recovery of the autoregression from
partner zeros, the quadrant rule, determinism, and the empty-output guard.
The full suite passes (219 tests, 1 skipped).

## E1: one, three or five steady states

**Step reading (7,056 couples).** Strict fixed points: one in 2,627 couples
(37.2%), two in 3,494 (49.5%), three in 935 (13.3%); never more than three, and
every strict fixed point is stable because the step pieces have zero slope.
Jump crossings make `n_completed = 2 n_fixed - 1` in **all 7,056 couples**, so
the completed count is 1, 3 or 5 exactly as the paper states, with the "two
unstable" states being the two jump crossings. A closed-form existence rule
(the NN, ZZ and PP candidates `u + v/(1-r)` must land in their own pieces)
agreed with the solver in 7,056/7,056 couples, and no fixed point fell on a
jump. The uninfluenced point (ZZ) exists for every couple in the grid; the
negative-negative point exists in 3,240 and the positive-positive in 2,124.
Restricting to negative plateau ≥ positive plateau (4,410 couples): one 1,547,
three 2,275, five 588.

Which regions give each count follows from the rule. Five requires both outer
points: `m_neg ≥ (1-r)(|K_neg| + u)` and `m_pos ≥ (1-r)(K_pos - u)` for both
spouses. At `u = 0` with shared inertia the five-state plateau pairs in the grid
are (5,8) and (8,8) for inertia .3 and .4, and (5,5), (5,8), (8,5), (8,8) for .5.
No couple with a plateau magnitude of 1 reaches five; (8,8) gives five in 420 of
441 couples, (5,8) in 289, (8,5) in 110, (5,5) in 56. Higher inertia widens the
region: five occurs in 64/784 couples at inertia (.3,.3) and 164/784 at (.5,.5),
while one occurs in 372/784 and 208/784 respectively.

**Smooth reading (1,488 couples).** Counts are always odd and stable minus
unstable is always one, consistent with index theory for a continuous map; the
even-count diagnostic was zero. At width 1.0: one in 666 (55.5%), three in 464
(38.7%), five in 70 (5.8%). At width 0.5: 64 / 68 / 12 of 144; at width 2.0:
120 / 24 / 0. The dense-grid recheck of 60 couples found **0 disagreements**.
The smooth count equals the step completed count in 94.4% of couples at width
0.5, 73.6% at width 1.0 and 54.2% at width 2.0: a wider ramp needs the outer
fixed point to sit on the saturated part of the sigmoid, so fewer couples keep
the outer states. The statement "one, three, or five steady states" therefore
reproduces under the smooth reading and, as 1/3/5 completed crossings, under
the step reading; the step map itself has at most three fixed points. The
paper's "in most of our data there was only one influenced steady state" is a
statement about their couples' estimates and is not testable here; the grid
only shows that a single steady state is the typical outcome when plateau
magnitudes are below roughly `(1-r)` times the thresholds (3-6 units).

The constructed test couple (ojive thresholds ±3 with plateaus ±6, repair of
4 at -8, damping of -4 at +8, inertia .5) has **five strict stable fixed
points**: (-12,-4), (-4,-12), (0,0), (4,12), (12,4). This illustrates that the
2003 trigger terms, not the 1999 ojive, are what can give the step map more
than three fixed points; it is not a published configuration.

## E2: published parameters

For every group and every one of the 400 plateau pairs the interior fixed point
is exactly the published uninfluenced pair, because both uninfluenced means lie
inside both thresholds and the ojive is zero there. Agreement with the published
influenced means within 0.5 per coordinate is therefore decided by the published
numbers themselves, not by the plateau:

| Group | Published uninfluenced (wife, husband) | Published influenced | Closest model fixed point | Max-norm distance | Pairs within 0.5 |
|---|---|---|---|---:|---:|
| Stable | 0.51, 0.72 | -0.65, 0.71 | 0.51, 0.72 | 1.16 | 0 / 400 |
| Divorced | -2.26, -1.89 | -1.95, -1.49 | -2.26, -1.89 | 0.40 | 400 / 400 |
| Happy stable | 1.07, 1.06 | -0.05, -0.33 | 1.07, 1.06 | 1.39 | 0 / 400 |
| Unhappy stable | -0.03, -0.41 | -1.88, 0.31 | -0.03, -0.41 | 1.85 | 0 / 400 |

The divorced match is trivial and the three failures are expected: Table 1
and Table 2 report **group means of per-couple estimates**, and a couple built
from mean parameters has the mean uninfluenced point as its interior attractor.
The published influenced means that differ from the uninfluenced means (notably
the stable wives' -0.65 versus 0.51, and the unhappy wives' -1.88 versus -0.03)
can only arise from couples whose attractor is an outer, influenced fixed point
or from averaging over heterogeneous couples; neither is recoverable from the
means. The number of fixed points across the sweep was one in 70-88, two in
196-204 and three in 108-132 of the 400 pairs per group.

Theoretical conversations from 625 initial conditions, 150 iterations, no clipping:

| Group | Plateaus (neg, pos) | Fixed points | Negative basin | Positive basin | Not converged |
|---|---|---|---:|---:|---:|
| Stable | 3, 3 | (0.51, 0.72) | 0.0% | 100.0% | 0.0% |
| Stable | 5, 3 | (-7.69,-7.61), (0.51,0.72) | 17.3% | 82.7% | 0.0% |
| Stable | 8, 5 | (-12.60,-12.61), (0.51,0.72), (8.71,9.05) | 33.3% | 66.7% | 11.7% |
| Stable | 8, 8 | (-12.60,-12.61), (0.51,0.72), (13.62,14.05) | 33.3% | 66.7% | 22.2% |
| Divorced | 3, 3 | (-2.26,-1.89) | 100.0% | 0.0% | 0.0% |
| Divorced | 5, 3 | (-10.88,-9.03), (-2.26,-1.89) | 100.0% | 0.0% | 0.0% |
| Divorced | 8, 5 | (-16.05,-13.32), (-2.26,-1.89) | 100.0% | 0.0% | 0.0% |
| Divorced | 8, 8 | (-16.05,-13.32), (-2.26,-1.89), (11.53,9.54) | 76.0% | 24.0% | 9.8% |
| Happy stable | 8, 8 | three | 33.0% | 67.0% | 21.3% |
| Unhappy stable | 8, 8 | three | 77.3% | 22.7% | 21.9% |

"Positive basin" counts final states with both scores above zero, which for the
stable groups is mostly the near-zero uninfluenced point (61.1% of initial
conditions at plateaus 8/5, versus 10.9% ending at the positive-positive
point). The qualitative published picture, that initial conditions decide the
end point and that divorced parameters give a negative attractor, holds for
every plateau pair; the exact fractions are properties of the declared grid and
plateaus, not published quantities. A post hoc check of the non-converged
orbits (not part of the frozen summary) found they are all **period-2 cycles**
straddling a threshold (139 of 625 for stable 8/8, 73 for stable 8/5, 61 for
divorced 8/8); the step map's basins are not exhaustive, and the papers do
not mention such cycles.

## E3: how well the published algorithm recovers known parameters

The exact rule "fit the autoregression where the partner scores zero" leaves
few points at 150 observations. Mean subset sizes per spouse were 7.6-51.5 at
noise SD 1 (large when the partner's uninfluenced state is near zero, 7.6 for
the divorced husband whose wife rarely scores exactly zero), 13.4-17.9 at SD 3,
and 9.0-10.5 at SD 5. The fit was impossible (fewer than three points) in at
most 0.5% of replications, except 2.0% for the divorced husband at SD 1.

| Quantity, SD 3 | Published rule, exact zeros | Published rule, `zero_tol` 1 | Joint bilinear, all points | Recursive LS, all points |
|---|---|---|---|---|
| Inertia bias (range over 8 spouse cells) | -0.019 to +0.001 | -0.023 to -0.010 | -0.021 to -0.012 | -0.034 to -0.021 |
| Inertia RMSE | 0.24-0.29 | 0.13-0.15 | 0.076-0.081 | 0.096-0.103 |
| Uninfluenced median abs error | 0.80-0.95 | 0.45-0.56 | 0.44-0.69 | 0.55-0.75 |
| Uninfluenced RMSE | 3.1-21.5 | 0.71-0.92 | 0.67-0.93 | 0.85-1.09 |
| Negative threshold bias (RMSE) | +0.3 to +2.0 (2.4-3.8) | -0.6 to +1.1 (1.5-2.5) | not estimated | not estimated |
| Positive threshold bias (RMSE) | -3.2 to -5.1 (4.2-6.0) | -2.0 to -3.8 (3.3-4.8) | not estimated | not estimated |
| Negative plateau bias (truth -3) | +0.4 to +1.0 | -0.02 to +0.5 | | |
| Positive plateau bias (truth +3) | -1.0 to -2.7 | -0.6 to -2.6 | | |

Bootstrap 95% intervals for every bias are in `summary.json`; those for the
published rule's inertia include zero in most SD 3 cells, so the rule is close
to unbiased for inertia and for the uninfluenced state but **imprecise**: an
inertia RMSE of about 0.26 is several times the published between-group
differences in inertia (0.03-0.12), and the uninfluenced state's RMSE is
dominated by replications whose inertia estimate approaches one (RMSE 179.6 in
one SD 5 cell against a median absolute error of 1.7). Threshold estimates are
biased toward zero whenever the partner rarely crosses the true thresholds: at
SD 1 the negative thresholds come out 3.2-5.8 units too close to zero and the
positive thresholds 5.4-7.0 too low, with plateau estimates near zero, because
the thresholds are not identified when the data never cross them and the
least-squares grid search then settles on noise. At SD 5 the negative
threshold bias shrinks to +0.1 to +1.6 but RMSE stays 3.2-4.3. The negative
side is recovered better than the positive side at every noise level because
the declared couples spend more time below their negative thresholds than above
their positive ones.

Admitting partner scores within ±1 as "zero" roughly triples the subset and
halves the inertia RMSE, at the cost of a small misspecification that is not
visible at these plateaus. The all-points joint bilinear fit has the lowest
inertia RMSE at every noise level but, being a bilinear fit to ojive data, is
biased in the uninfluenced state for groups whose uninfluenced state is away
from zero (+0.58 and +0.70 for the divorced spouses, +0.43 and +0.27 for the
unhappy stable spouses at SD 3); the recursive identifier behaves like the
joint fit with extra variance from its forgetting factor. These are properties
of the estimators under a declared noise model and sample size. They do not
measure the original data's noise, do not reproduce or contradict any
published parameter value, and are offered as a methods check rather than a
criticism of the original analyses, whose group means over 108 and 17 couples
average much of the per-couple imprecision but not the threshold bias.

## E4: repair and damping

Baseline bilinear couples at slopes (.5,.3): stable (1.15, 1.29) and happy
stable (2.05, 2.05) have one stable positive-positive fixed point; divorced
(-10.12, -9.12) and unhappy stable (-1.08, -1.24) have one stable
negative-negative point. Slopes (.3,.2) give the same pattern with points
nearer zero. Slopes (.7,.4) make the negative-negative Jacobian's spectral
radius exceed one: stable and happy stable gain an unstable saddle, while
divorced and unhappy stable have **no fixed point and escalate without bound**.

Across the **360 damping configurations, damping created a positive-positive
stable steady state in 0 and destroyed one in 0**. Across the 336 repair
configurations, repair destroyed the negative-negative attractor in 28 (all
divorced at slopes .5/.3, whose attractor lies below both repair thresholds) and
created none; the orbit then becomes a bounded cycle of period 5, 15, 23 or 29
in the negative quadrant (wife -7.0 to -4.0, husband -5.9 to -1.6), not a
positive attractor. At slopes (.3,.2) the divorced attractor (-4.16, -3.67)
sits above the repair thresholds and is untouched. At slopes (.7,.4) repair
introduces fixed points for the divorced and unhappy stable couples, none of
them stable; whether those orbits stay bounded was not checked. The
only new stable states anywhere are three mixed-quadrant points created by
repair 3.99 together with damping -5.03 at threshold 4 at slopes (.3,.2):
(-6.36, 4.19) for the stable couple and (-5.36, 4.90), (4.78, -5.51) for the
happy stable couple.

The negative damping result is not a sampling accident. With influence
functions that are non-decreasing in the partner's score (bilinear with
non-negative slopes, or an ojive with negative plateau ≤ 0 ≤ positive plateau)
the undamped map `F` is order-preserving, and constant-trigger damping gives a
map `F' ≤ F` pointwise. If `F'` has a fixed point `x*`, then `F(x*) ≥ x*`, and
Tarski's fixed-point theorem gives `F` a fixed point at or above `x*`, hence in
the same positive quadrant; with bilinear pieces the slopes, and therefore the
stability, are unchanged. So under the module's reading of the 2003 verbal
definition ("a standard trigger function that makes the time series jump ...
with amplitude E at the trigger threshold K"), damping can move or remove a
positive-positive attractor but never create one. **The 2003 statement did not
reproduce under this formulation.** It may hold under the authors' actual
algorithm in Gottman, Murray, Swanson, Tyson and Swanson (2002), for example if
damping reduces the positive slope rather than subtracting a constant, or in
combination with repair as the mixed-quadrant cases show; that book was not
consulted and a slope-reduction form is not implemented in the module, so the
claim remains untested rather than refuted.

## What reproduced and what did not

| Published statement | Outcome |
|---|---|
| Sigmoidal influence gives one, three or five steady states | Reproduced under the smooth reading (counts always 1, 3 or 5; 0 even counts in 1,488 couples) and as completed null-cline crossings under the step reading (7,056/7,056 couples). The step map itself has at most three fixed points, all stable; its "unstable" states are jump crossings, not fixed points. |
| Three stable and two unstable states in the five-state case | Reproduced: stable minus unstable equals one in every smooth couple. |
| Uninfluenced steady state is `a/(1-r)` | Reproduced exactly (unit test and 1,600 sweep rows). |
| Published influenced steady-state means | Not matched for stable, happy stable or unhappy stable (distance 1.16-1.85); matched trivially for divorced (0.40). Expected, since the published values are means of per-couple estimates. |
| Initial conditions decide the end point; divorced parameters attract negatively | Reproduced qualitatively; 24% positive endings for divorced only at plateaus 8/8. Period-2 threshold cycles, not mentioned in the papers, capture 10-22% of initial conditions at large plateaus. |
| The published estimation algorithm | Runs as described and recovers autoregression exactly on noiseless data with partner zeros. At 150 points with SD 3 process noise it is near unbiased but imprecise for inertia (RMSE 0.24-0.29) and the uninfluenced state, and biased toward zero for thresholds (positive threshold -3.2 to -5.1). |
| Damping may create a positive-positive stable steady state where none existed | Not reproduced in 0/360 configurations; provably impossible for constant-trigger damping with monotone influence. The authors' exact damping algorithm was not available to test. |

## Observations about the model module

No blocking defect was found in `regesh/bonding/dyad.py`, and the module was
not edited by this study. Four behaviours matter for interpretation. First,
`steady_states` silently drops a fixed point that lands exactly on a jump of
the step map; integer grids with integer thresholds can produce such
coincidences (for example `u = 0`, `r = .5`, plateau -3, threshold -6), which is
why this study used the non-integer published thresholds and flags degenerate
couples (0 occurred). Second, `fit_published_algorithm` resolves ties in its
threshold grid search by grid order and assigns a plateau of zero when no
residual lies beyond a candidate threshold, so thresholds are returned even when
the data never cross them; that is the mechanism behind the E3 threshold bias
at low noise, and consumers should check the fitted plateau values. Third, the
uninfluenced state `c/(1-r)` is reported without any guard against inertia
estimates near one, which produces the heavy tails seen in E3. Fourth,
`steady_states` dispatches smooth specs to `fixed_points_numeric` with its
25×25 default seed grid, which costs 0.3-1 s per couple; coarser seeding is
much faster but a single 9×9 grid missed roots in 4 of 150 pilot couples,
whereas the union of two offset coarse grids missed none in 210.

These simulations establish properties of a published difference-equation
model and of a published estimation rule. They establish nothing about the
emotional lives of couples, about persistent affect in software agents, or
about the Regesh controllers, none of which were run.
