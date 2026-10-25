# Homeostatic reinforcement learning: independent Figure 6 reproduction

Paper: [Keramati and Gutkin (2014), eLife 3:e04811](https://doi.org/10.7554/eLife.04811), a peer-reviewed Version of Record published 2 December 2014. Search and source inspection: 25 October 2026. The [original Figure 6 parameter supplement](https://cdn.elifesciences.org/articles/04811/elife-04811-fig6-data1-v1.docx) was downloaded and parsed directly. DOI links, retrieved byte sizes and SHA256 hashes are in [homeostatic_sources.json](homeostatic_sources.json). The article and supplements are available under Creative Commons Attribution; credit the authors when redistributing their material. No authors' executable implementation or random seed was located in the bounded paper/supplement/web search.

The reproduced experiment is the paper's Figure 6, concerning associative regulation of an internal variable. There are two actions, increasing or decreasing the variable by one unit. Equation 1 gives drive `D(H)=|H|^(4/3)` at setpoint zero; Equation 2 gives reward `D(H)-D(H_next)`. The original parameters are learning rate 0.4, beta 0.05, discount 0.9 and initial state 30. The paper uses 100,000 agents over 1,500 actions, plus a one-agent million-action trajectory. Its text reports a maximum deviation of 61 in the population and no deviation above 40 in the long single trajectory. Final means and standard deviations are curves, with no precise tabulated endpoint; none was invented or digitized here.

This is an **independent method reimplementation**, not an exact replay of authors' code or a new animal experiment. Standard tabular Q-learning starts with all Q values zero. The original supplement names beta an exploration rate but does not state the softmax convention. Before the first simulation, we fixed inverse-beta softmax as the primary assumption and the opposite temperature convention as a sensitivity condition. Their relative agreement with the paper must not be used to retrospectively select the convention.

The implementation retains the full published population and horizon, generated as 32 independent batches of 3,125 agents. Each agent has its own Q table; batches have distinct declared PCG64 seeds. A paired no-learning condition uses the same random variates while choosing actions with probability one half. The temperature sensitivity uses the same declared seeds and population. An additional 32-agent vectorized experiment retains the million-action horizon, using independent table rows and draws from one declared joint random stream. Its visit histogram aggregates all 32 agents; it must not be called the paper's one-agent histogram. Individual maximum deviations are saved separately.

The allocation guard raises if a state reaches absolute value 256. It never clips, reflects, resets, drops, or censors an agent. Thus a finite simulation-table boundary cannot silently manufacture the reported stability. The control is a genuine unconstrained random walk up to the same fail-fast guard. No LLM, human label, API charge or main Regesh controller is involved.

Run from the repository root into a **new, empty output directory**:

```sh
OPENBLAS_NUM_THREADS=1 python -m regesh.reproductions.homeostatic \
  --config configs/reproductions/homeostatic.json \
  --sources docs/reproductions/homeostatic_sources.json \
  --output /tmp/homeostatic-reproduction
python -m unittest tests.test_homeostatic_reproduction -v
python -m regesh.reproductions.homeostatic_analysis \
  --run /tmp/homeostatic-reproduction \
  --output /tmp/homeostatic-posthoc-analysis.json
```

The runner requires the fixed configuration hash in the prospective source record. It does not depend on the download cache: the independently transcribed numerical parameters are versioned locally. Original-source URLs and checksums permit a separate source audit. To run a modified experiment, copy the configuration and create a separately labelled source record with its new hash; do not overwrite the completed study.

Raw NPZ files contain per-agent terminal states and maximum deviations, integer visit counts, every-time population moments, and four example histories per cohort batch. `cohort_*_moments.npz` columns are step, mean, mean square, mean absolute deviation and **mean of batch maxima**; that last column is not the population maximum. `summary.json` computes the actual population maximum separately. Normal 95% intervals use the 32 independent batch means, not correlated time samples. Raw batch rows, executed source and hashes are retained alongside the summary. Maxima are descriptive stochastic realizations, not deterministic success thresholds or confidence intervals.

## Measured cohort results

Each row contains 100,000 independently adapting agents over 1,500 actions; the no-learning row has no adapting values.

| Condition | Final mean state | Final state SD | Final mean absolute deviation | Maximum absolute state over history |
| --- | ---: | ---: | ---: | ---: |
| Paper parameters, **assumed inverse-beta convention** | 0.04668 | 6.93353 | 5.41912 | 86 |
| Paired no-learning random walk | 30.19544 | 38.65094 | 39.79792 | 192 |
| Opposite, temperature-beta sensitivity | 0 | 0.00894 | 0.00004 | 46 |

The learned model regulates the internal variable relative to the paired random-walk control. This expected benefit of regulatory learning is not evidence that emotion outperforms an ordinary adaptive controller. Neither comparator is the frozen Regesh hybrid or ordinary controller, and no software-design score was measured here.

The primary-minus-control mean absolute deviation difference is **−34.3788**, normal 95% interval **[−34.565535, −34.192065]**, calculated across the 32 paired independent batches. The relative reduction is 86.38%. The cohort mean absolute deviations' respective normal intervals are [5.393856, 5.444384] and [39.605944, 39.989896]. These are new simulation comparisons with a no-learning control, not effect sizes reported in the original paper.

The primary cohort's extreme value **does not reproduce the reported maximum of 61**: its maximum is 86, and a post hoc raw-data audit finds 86 of the 100,000 agents exceeded 61. This is an unresolved tail/protocol discrepancy. Unknown authors' code, seed and beta convention prevent attribution to one cause; random seed alone is not established as an explanation. Broad agreement of the mean trajectory with regulation does not resolve this mismatch.

The temperature sensitivity's near-zero endpoint requires care: with only unit increase/decrease actions, even and odd event times have different distributions. Near-zero state at action 1,500 does not imply the state remains at zero at every action. A separately labelled post hoc audit averages the last 100 action times to include both phases. This diagnostic does not change the primary endpoint or select a preferred beta convention.

That last-100-action mean absolute deviation is 5.441529 for the primary model, 39.410308 for the no-learning control and 0.500034 for the temperature sensitivity. The sensitivity's endpoint value of 0.00004 should therefore never be interpreted as an all-time equilibrium deviation of that size.

Across the **32 million-action trajectories**, individual maximum deviations range from **32 to 50**. Of the 32 replicas, **21 (65.625%)** have a maximum no greater than the paper's reported 40; 11 exceed it. The paper supplies one such realization, so neither an exact-match demand nor a claimed equivalent maximum distribution is justified. The reported long-run visit count is 32,000,000, and its saved histogram aggregates those 32 trajectories.

The complete run executed **482 million state transitions** in **459.41 seconds wall time** while sharing CPU with other experiments; this is not a benchmark of isolated runtime. Eight targeted tests passed. A separate analysis verifies every original artifact hash and independently recomputes terminal means, standard deviations, absolute deviations and maximum values from raw individual records using Python's standard library; all match the recorded primary summary. The source and output hashes are in `manifest.json`; post hoc diagnostics are separately retained in `analysis.json` with their own source and input-manifest hashes. No allocation guard was reached, and no trial was omitted.

This experiment tests simulated homeostatic learning. It cannot establish persistent LLM affect, biological fidelity, subjective experience or improved useful idea generation. In particular, a finite observed maximum cannot establish an absolute long-run safety boundary. With finite Q values and nonzero softmax probabilities, arbitrarily long finite action sequences remain possible.
