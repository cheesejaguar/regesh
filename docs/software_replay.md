# Exact recorded-response software replay

`regesh.software_replay` reconstructs the actor trajectory from saved model responses and **recorded public tool feedback**. It does not ask a model to regenerate an answer. Generation seeds and identical prompts did not guarantee identical fresh responses in the CPU model experiment; exact recorded-response replay is therefore a different, stronger trajectory-closure check than another live decode.

```bash
python -m regesh.software_replay \
  --run results/software_real --weights results/benchmark \
  --output results/software-replay-new --recheck
```

Use a new output directory. JSON/JSONL records, source snapshots and weights may be plain files or lossless gzip archives. The CLI checks archived source manifests and exact trained-weight hashes. It loads the repository-owned evaluator source captured at execution, so a later report-rendering change does not silently alter the actor replay. Installed task definitions must still match the frozen task hash; a mismatch aborts. Do not treat arbitrary external source snapshots as trusted repository code.

For every trial, replay reconstructs the complete initial `TrialState`, recomputes action predictions/scores and strategies, regenerates and compares every exact prompt, checks generation seeds/model identity/temperature/token ceilings, consumes the recorded response, repeats source/schema parsing, and applies only the recorded public diagnostic feedback. Every resulting adaptive state is compared to the saved state. A record claiming hidden feedback as public is rejected; public input identities must match the original public examples and diagnostic case. Final code, errors, usage totals, weight references, call/tool counts and derived score consistency are also checked.

No candidate code executes and no independent hidden checker runs while actor contexts are reconstructed. The fixture tests explicitly prohibit both provider calls and subprocess execution in that phase, and verify rejection of corrupted prompts, adaptive state and source manifests.

`--recheck` begins only **after all actor trajectories close**. It separately executes the frozen condition-blind checker on each stored initial and final program. Pass/fail, typed correctness, errors, dynamic line counts, static measurements and other stable outputs must match. Wall timing and allocator-dependent peak figures are excluded from equality. These are additional evaluator-only calls; they provide no feedback to a controller and consume no model calls.

## Completed measurement

The actual `results/software_real` archive was replayed with `--recheck`. [Summary](../results/software_replay/summary.json):

- **24 actor trials and 48 generation phases verified**, covering hybrid and ordinary adaptive conditions.
- **0 model calls, 0 live-provider calls, 0 network calls** during replay.
- **0 hidden-checker calls during actor reconstruction**; 24 recorded public tool results consumed.
- Afterwards, **48 independent executable rechecks** (initial and final for each trial), with **all stable functional/error/line results matching** and no mismatches.

This establishes recorded-response trajectory closure and executable grading reproducibility for this archive. It does not establish deterministic fresh generation, cross-version model reproducibility, new behavioral benefit, psychological traits or subjective experience. The original real-model result and its uncertainty remain authoritative; replay adds no new solution-generation observations.
