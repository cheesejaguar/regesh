# Procedural search experiment

These are executed synthetic search results, not measurements of a live language model.

| Condition | Quality | Constraint compliance | Useful novel fraction | Included-mechanism diversity |
|---|---:|---:|---:|---:|
| memory_planning | 0.5668 | 0.7792 | 0.6333 | 9.33 |
| static_persona | 0.5754 | 0.7826 | 0.6667 | 11.43 |
| turn_sentiment | 0.6218 | 0.8820 | 0.6500 | 12.65 |
| heuristic | 0.5949 | 0.8581 | 0.6667 | 7.10 |
| hybrid | 0.6130 | 0.9177 | 0.8167 | 7.48 |
| ordinary_adaptive | 0.6258 | 0.9036 | 0.7000 | 7.60 |
| unlabelled_hybrid | 0.6130 | 0.9177 | 0.8167 | 7.48 |
| frozen | 0.5297 | 0.7724 | 0.6833 | 2.22 |
| shuffled | 0.6322 | 0.9055 | 0.7833 | 10.60 |
| no_memory | 0.6133 | 0.9177 | 0.8167 | 7.47 |
| no_counterfactual | 0.6311 | 0.9203 | 0.7667 | 11.43 |

Differences are hybrid minus comparator; 95% paired seed-cluster bootstrap intervals.

| Comparator | Difference | 95% interval |
|---|---:|---|
| memory_planning | 0.0462 | [0.0098, 0.0847] |
| static_persona | 0.0376 | [-0.0028, 0.0826] |
| turn_sentiment | -0.0087 | [-0.0336, 0.0183] |
| heuristic | 0.0181 | [-0.0167, 0.0587] |
| hybrid | 0.0000 | [0.0000, 0.0000] |
| ordinary_adaptive | -0.0128 | [-0.0370, 0.0094] |
| unlabelled_hybrid | 0.0000 | [0.0000, 0.0000] |
| frozen | 0.0833 | [0.0311, 0.1425] |
| shuffled | -0.0192 | [-0.0443, 0.0084] |
| no_memory | -0.0003 | [-0.0009, 0.0000] |
| no_counterfactual | -0.0181 | [-0.0443, 0.0058] |

The unlabelled hybrid is exactly the same computation. Any identical performance is expected; labels are not evidence of an emotion-specific benefit.
Static persona and sentiment conditions are numerical analogues only; they do not test prompt conditioning on an LLM.
Originality means absence from four explicitly listed canonical designs. It does not establish published novelty, and distance is not used as a creativity score.
The held-out scientific family is a modest formula transfer test; all families share a six-knob design grammar.
Mechanism diversity counts distinct subsets of enabled knobs (64 possible), while canonical design diversity includes their levels (4096 possible). The four-design originality corpus has weak coverage.
Brier is common outcome-model calibration on policy-selected actions, not independent policy calibration. Internal burden is a regulator cost, not an externally enforced resource constraint.
Structural corrections followed an exploratory pilot whose raw files were overwritten; see docs/exploratory_pilot.md. All results are exploratory. Coefficients were not optimized against held-out scores; intervals are not multiplicity-corrected.
See docs/benchmark_protocol.md for the equations, limitations, and live-model follow-up.
