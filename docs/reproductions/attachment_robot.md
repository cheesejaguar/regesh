# Hiolle, Lewis and Cañamero (2014): arousal regulation and affective adaptation in a simulated robot

Execution date: **2026-10-25 UTC**. Primary source: [Arousal regulation and
affective adaptation to human responsiveness by a robot that explores and
learns a novel environment](https://doi.org/10.3389/fnbot.2014.00017),
*Frontiers in Neurorobotics* 8:17, a peer-reviewed article published under
Creative Commons Attribution. This study simulates the paper's three
experiments (sections 3.1-3.3; Figures 6, 8 and 9). It is not a rerun with the
authors' Nao robot, camera, neural networks or human experimenter, and it is
not an evaluation of any Regesh controller.

## Scope and fidelity

The paper's equations (1) and (3)-(7) and the Table 1/2 parameters are
imported unchanged from `regesh/bonding/attachment.py`; the runner
(`regesh/reproductions/attachment_robot.py`) does not re-implement them and
copies both source files into the output with their hashes. No authors' code
was located; none was copied or executed. The article text was read in full;
Algorithm 1 and the figures are images and could not be read, so the per-step
order of the architecture is declared in the runner docstring.

**Fidelity label: independent simulation with declared substitutes.** The
original stimulation signal (Hopfield recall error plus Kohonen weight change
over camera contours, Eq. 2) is unavailable and is replaced by a per-position
familiarity decay. The Nao's lateral stepping and the automated caregiver are
simulated. Therefore **absolute seconds, request counts and learning totals are
not comparable with the paper**; only directions and ratios are compared, and
even those depend on the substitutes listed below.

## Fixed protocol

The [configuration](../../configs/reproductions/attachment_robot.json) was
frozen at 23:35:55 UTC before any comparison seed ran. Its substitutes and
constants are:

| Element | Paper | Simulation (declared) |
|---|---|---|
| Timestep | about 300 ms loop; step 160 is about 50 s | dt = 1/3 s, so the 1 s caregiver latency is exactly 3 steps |
| Environment | 7 lateral views of a table with few (simple) or many (complex) objects | 7 positions; per position 2-3 (simple) or 6-8 (complex) objects drawn per seed |
| Stimulation | Eq. 2 recall error and map weight change | `s_env * exp(-f_p / (lambda * n_p)) + N(0, .02^2)`, clipped to [0, 1]; s_env .85 (simple) / 1.0 (complex), lambda 2; `f_p` counts attending steps at position p and persists across revisits |
| Movement | one Nao sidestep per position, duration unstated | a sidestep occupies 6 steps (2 s) and completes once started; the destination view is perceived in transit without learning |
| Behaviour | Figure 4 winner-take-all on arousal | `behavior(arousal, .6, .4)`: request / learn / explore; a request is counted when the behaviour switches into request |
| Caregiver | automated response 1 s after Find-Human starts; duration unstated | face and head contact set to 1 for 3 steps beginning 3 steps after a request onset; non-responsive control never responds |
| Eq. 4 condition | `Ch(t) > 0 or Fh(t) > 0` on Eq. 1 traces | evaluated on the raw sensors (traces never reach zero; see ambiguities); literal trace reading kept as a sensitivity |
| Eq. 5 timing | "when the behavior Find-Human is active" | steps whose selected behaviour is request ("literal", primary); "episode" sensitivity also counts the intervention steps |
| Experiment 3 human | experimenter responsive, stopped after 150 s, run about 350 s | responsive until 150 s, then non-responsive, 350 s horizon; intervention 1 s (primary) or 3 s (sensitivity) |
| Horizon | representative runs 150-170 s | Experiments 1-2 stop when the table is traversed twice (1 to 7 and back) or at 600 s |

The stimulation constants were fixed on a separate pilot seed (29000) with one
criterion only: the independent profile's arousal must visit all three bands in
both environments with the responsive caregiver. The first candidate met it
(simple .825/.100/.075 and complex .821/.090/.090 explore/learn/request) and no
other candidate was tried; the runner re-executes the pilot and records it in
`summary.json`. Published endpoints were not used for calibration.

Cells: 2 environments × 2 profiles × {responsive, non-responsive} with the
paper's 10 runs per cell (seeds 30000-30009) plus 100 bootstrap seeds
(31000-31099); sensitivities with 1 s sidesteps and with the literal trace
condition (100 seeds each, responsive); and four adaptive cells (literal or
episode semantics × 1 s or 3 s intervention) with the paper's 5 runs plus 100
seeds. Seeds sharing an environment share the object layout and the full noise
vector, so within-environment profile contrasts are paired by seed; the
complex-versus-simple contrasts resample the two groups independently.
Intervals are nominal 95% percentile bootstraps with 10,000 resamples and no
multiplicity correction.

```bash
OPENBLAS_NUM_THREADS=1 python -m regesh.reproductions.attachment_robot \
  --config configs/reproductions/attachment_robot.json \
  --output results/papers/attachment_robot-new
python -m unittest tests.test_attachment_robot_reproduction -v
```

The output directory must be empty. Outputs: `config.json`, `source.py`,
`equations_source.py`, `runs.jsonl.gz` (all 2,100 run records),
`series_<cell>.npz` (one representative time series per cell, the first
declared seed), `summary.json` and `manifest.json` with SHA-256 hashes of
every file.

## Declared ambiguities

1. **Stimulation substitute.** Absolute pace and learning are determined by
   `s_env`, `lambda` and object counts, none of which exist in the paper.
2. **Eq. 4 is inconsistent with Eq. 1 as printed.** Eq. 1 traces decay
   geometrically and never reach zero, so the literal condition `Ch > 0 or
   Fh > 0` would keep the window-average branch forever after the first
   contact and `beta_h` would never act, contradicting the paper's description
   of the two profiles. The primary reading tests the raw sensors; under it
   `alpha_sensor` affects only the recorded traces, not comfort.
3. **Eq. 5 timing.** Comfort removes arousal from the request band almost
   immediately (Eq. 3 subtracts `alpha_ar * Comf`), so with the literal reading
   an answered request yields about three decrease steps during the latency
   and at most one increase. The published `(1 - Resp)` decrease term is kept
   verbatim, as the module documents.
4. **Arousal floor.** The paper states none; the module floors at zero.
5. **Adaptive initial parameters.** Table 2 lists `beta_h = .7`, `tau_h = 3`
   as initial values, but Eqs. 6-7 give .85 and 6.5 at `Resp(0) = .5`; the
   equations are applied from the start.
6. **Learning proxy.** Familiarity increments only while attending (learn or
   request); the moving camera view is not learned in transit.
7. **Movement duration.** With one position per timestep the independent
   profile would cross the table several times inside one comfort window,
   which the paper's temperature plots exclude; 2 s per sidestep was chosen for
   physical plausibility, with 1 s as a sensitivity.
8. Start state (position 1, arousal 0, so the robot steps before learning
   position 1), run termination, PCG64 streams and per-step order are declared
   in the configuration and runner.

## Measured results

The full study executed **2,100 runs, 601,324 timesteps (200,441 simulated
robot-seconds) in 5.1 s wall time**. All cells completed both traversals
before 600 s; no run was censored. Seeds change only the object counts and the
.02-SD noise, so most endpoints are identical across seeds (request counts have
zero variance in every primary cell; two-traversal times vary by 0-2.3 s). The
intervals below therefore describe the declared noise, not the paper's
run-to-run variability, although the paper also reported "very similar" runs.

### Experiments 1 and 2: fixed profiles (100 seeds per cell)

| Cell | Requests | Requests / min | Mean arousal | Explore / learn / request fraction | Learning steps | Positions learned | End reached (s) | Two traversals (s) | Comf > .1 span (s) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Simple, needy, responsive | 4.00 | 7.59 | .283 | .588 / .294 / .118 | 22.8 | 6 | 15.7 | **31.6** | 3.00 |
| Simple, independent, responsive | 2.00 | 4.50 | .120 | .825 / .103 / .072 | 8.0 | 2 | 13.3 | **26.7** | **12.33** |
| Simple, either profile, non-responsive | 6.00 | 8.74 | .455 | .332 / .379 / .289 | 51.6 | 6 | 26.3 | 41.2 | none |
| Complex, needy, responsive | 5.00 | 9.60 | .272 | .596 / .243 / .160 | 22.2 | 5 | 16.1 | **31.4** | 3.00 |
| Complex, independent, responsive | 2.00 | 4.62 | .130 | .821 / .090 / .090 | 6.0 | 2 | 13.0 | **26.0** | **12.33** |
| Complex, either profile, non-responsive | 7.00 | 6.83 | .571 | .211 / .305 / .484 | 112.5 | 6 | 43.3 | 61.5 | none |

The 10-seed cells that match the paper's run count give the same means
(for example simple independent two traversals 26.67 s in all 10 runs; complex
needy 31.17 s, range 29.7-34.3). Without comfort the two Table 1 profiles are
identical run for run, because they differ only in how comfort is processed.

Paired needy-minus-independent contrasts with the responsive caregiver:

| Endpoint | Simple: difference [95% CI] | Complex: difference [95% CI] |
|---|---:|---:|
| Requests | **+2.00 [2.00, 2.00]** | **+3.00 [3.00, 3.00]** |
| Requests per minute | +3.09 [3.07, 3.12] | +4.99 [4.86, 5.11] |
| Mean arousal | +.164 [.162, .166] | +.143 [.140, .146] |
| Learn-band fraction | **+.191 [.188, .195]** | **+.154 [.143, .165]** |
| Explore-band fraction | **−.238 [−.241, −.235]** | **−.224 [−.233, −.216]** |
| Learning steps | +14.8 [14.5, 15.2] | +16.2 [14.9, 17.5] |
| Two traversals (s) | +4.95 [4.85, 5.05]; ratio **1.186** | +5.39 [4.96, 5.84]; ratio **1.207** |
| Comf > .1 span (s) | −9.33 | −9.33 |

Complex-minus-simple requests: needy **+1.00 [1.00, 1.00]**, independent
**0.00 [0.00, 0.00]**. Responsive-minus-non-responsive within profile: fewer
requests (−2 needy, −4 or −5 independent), 29-106 fewer learning steps and
10-35 s faster double traversals.

### Comparison with the published observations

| Published observation (quoted) | Measured | Status |
|---|---|---|
| Needy requests comfort "much more often" (both environments) | +2 and +3 requests, intervals exclude zero | Direction reproduced |
| Independent Comf "trace lasting up to 10 s" | 12.33 s above the .1 floor after the first response (every independent run ends inside its second span) | Order of magnitude consistent; an arithmetic consequence of `beta_h = .95` and the 3-step intervention at dt = 1/3 s, not an independent observation |
| Needy shows more medium-arousal episodes; "the independent robot explored more, the needy robot learned more" | learn fraction +.19/+.15, explore fraction −.24/−.22, learning steps +15/+16 | Direction reproduced |
| Complex environment: independent 140 s, needy 170 s for two traversals (ratio 1.21) | 26.0 s versus 31.4 s, ratio 1.207, paired +5.39 s | Direction and ratio agree; **absolute seconds do not** (about one fifth of the published durations) |
| Simple environment: "both profiles go through the setup at a similar pace" (end about 50 s, back about 83 s) | 26.7 s versus 31.6 s, ratio 1.186, paired +4.95 s [4.85, 5.05] | **Not reproduced in substance.** The prespecified rule (gap closer to 1 than in the complex environment) is met only marginally (1.186 versus 1.207); the needy is clearly slower in both environments here |
| Independent "did not solicit the attention of the caregiver more often" in the complex environment; needy "produced more regulatory behaviors" | independent 0.00, needy +1.00 | Direction reproduced |

The absolute pace discrepancy has an identified cause in the model rather than
in the data: with Table 1 parameters, Eq. 3 subtracts `.6 * Comf` per step, so
arousal reaches the zero floor within two steps of a response and stays there
while `Comf > .1`. The independent robot therefore steps for 12 s (six
positions, the whole table) after each of its two requests, and the needy for
3 s; over the whole run the independent robot attends to only two of the seven
positions. The paper's robot took 50 s per traversal, which the substituted
stimulation and declared step duration cannot and were not tuned to recover.

### Experiment 3: adaptive regulation (100 seeds per cell)

| Cell | Resp at 150 s (mean) | Resp(150) − .5 [95% CI] | Resp at end | Resp(end) − Resp(150) [95% CI] | beta_h / tau_h at 150 s | beta_h / tau_h at end | Requests phase 1 / 2 |
|---|---:|---:|---:|---:|---|---|---:|
| **Literal, 1 s (primary)** | **.014** | **−.486 [−.486, −.485]** | .000 | −.014 [−.015, −.014] | .996 / 9.90 | 1.000 / 10.00 | 8.0 / 5.4 |
| Literal, 3 s | .051 | −.449 [−.453, −.444] | .000 | −.051 [−.056, −.047] | .985 / 9.64 | 1.000 / 10.00 | 8.0 / 5.4 |
| Episode, 1 s | .532 | +.032 [.029, .035] | .271 | −.261 [−.282, −.241] | .840 / 6.27 | .919 / 8.10 | 18.6 / 1.9 |
| Episode, 3 s | .977 | +.476 [.476, .477] | .916 | −.060 [−.063, −.058] | .707 / 3.16 | .725 / 3.59 | 17.0 / 3.9 |

The published pattern is "responsiveness steadily increased (approximately from
second 5-140)" under the responsive experimenter, then "decreased" at the
requests near 170 and 240 s, with `beta_h`/`tau_h` moving toward the needy
profile and back toward the independent one.

**Under the primary literal reading the first half does not reproduce and runs
in the opposite direction.** Responsiveness fell from .5 to .47 at the first
request (which the paper also describes: "when a request was made, the
evaluated responsiveness started decreasing since the caregiver had not yet
responded") and kept falling at every answered request, reaching .014 by 100 s
in every run; `beta_h` rose to .996 and `tau_h` to 9.9, the independent end of
the range. The reason is analytic: each request produces three decrease steps
during the 1 s latency and at most one increase step, because comfort drops
arousal below `HighThresh` on the first comfort step. A second published-equation
effect then appeared: as `Resp` approaches zero, Eq. 6 gives `beta_h` close to
1, so comfort stops decaying; in the representative run `Comf` stayed above .1
for 137 s, arousal was pinned at zero and no request occurred between 82 s and
225 s. The phase-2 "decrease" in this cell is a floor effect of .014.

Under the **episode sensitivity**, in which Find-Human stays active while the
human intervenes (the consummatory gaze behaviour of Figure 4), the direction
depends on the intervention length. With 1 s interventions the three decrease
and three increase steps cancel to first order and `Resp` drifts to .532 by
150 s, then falls to .271 after two unanswered requests. With 3 s interventions
`Resp` rises to .977 (`beta_h` .707, `tau_h` 3.2, the needy end) and then falls
to .916 after 150 s. Both reproduce the published rise-then-fall direction, but
they required two undeclared-in-paper choices, and the fall after 150 s is
small because the robot has learned the environment by then and makes few
requests (1.9-3.9 in phase 2, versus the paper's two). These are sensitivities,
not the prespecified primary result.

### Sensitivities for Experiments 1-2

With **1 s sidesteps** the needy-minus-independent directions persist (simple:
requests +.41 [.31, .51], learn fraction +.123, two traversals +3.3 s, ratio
1.166; complex: requests +2.00, ratio 1.214), with 14-23 s double traversals;
the independent runs end inside their single comfort span, so no completed
span duration is recorded for them.

With the **literal trace condition in Eq. 4**, the needy comfort span becomes
11.67 s and the independent 14.67 s (versus 3.00 and 12.33 s), both profiles
make exactly 2 requests in both environments, and the two-traversal ratio is
.899 (simple) and 1.029 (complex). The profile contrast collapses or reverses,
which is why the primary reading evaluates the condition on the raw sensors.

## Tests and verification

Seven unit tests check: Eq. 1, 3, 4, 5, 6 and 7 against hand calculations with
Table 1/2 parameters (including the comfort floor, the arousal floor, the
`(1 - Resp)` decrease and the stimulation substitute); same-seed determinism of
records and time series and shared layouts across profiles; the caregiver
latency of exactly 3 steps (1 s) with a 3-step intervention and Eq. 1 traces
of 1 then .9; needy requesting at least as often as independent on three seeds;
zero comfort, zero responses and identical profile runs under the
non-responsive caregiver; collapse of `beta_h` under the literal trace
condition; and that `run()` writes a manifest whose hashes close over every
output file, records the pilot and refuses a non-empty directory. The manifest
hashes of the delivered run were re-verified after execution, including the
configuration, runner and equation-module hashes.

## What this does and does not show

Under the imported equations and the declared substitutes, the needy profile
requests more comfort, spends more time in the learning band, learns more and
traverses the table more slowly than the independent profile, and the
independent profile's comfort lasts about four times longer. These directions
match the paper's prose. The absolute durations do not, the simple-environment
"similar pace" claim does not, and the adaptive responsiveness increase does
not reproduce under the literal reading of Eq. 5; it appears only when the
regulatory behaviour is assumed to persist through a comfort episode longer
than the response latency. The runs are nearly deterministic, so the narrow
intervals do not measure the variability of the original robot. None of this
establishes attachment, affect, biological fidelity or a benefit for any
Regesh controller.
