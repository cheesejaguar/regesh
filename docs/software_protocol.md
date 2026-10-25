# Frozen-controller same-model executable software design

This protocol was written before inspecting real-model task results. The runner is `regesh/software_eval.py`; task specifications, independent oracles and fixed generated cases are `regesh/software_tasks.py`. `configs/software_experiment.json` records the prospective default budget. Its status becomes a completed measurement only when a real model adapter runs it; fixture results validate the checker, not model performance.

## Hypothesis and transfer boundary

Primary question: does the frozen learned hybrid produce a larger fraction of correct hidden executable tests than a frozen ordinary adaptive controller when both regulate the **same real model** under identical call, tool and completion-token ceilings?

Both shared models are loaded unchanged from the prior procedural experiment. The hybrid predicts four consequences of four cognitive actions and scores projected internal costs; ordinary adaptation has 288 active nonlinear direct-value coefficients. Both receive the identical 17 context measurements and choose from divergence, refinement, combination and verification. Generic weights are never fitted or updated on these tasks. Within-trial progress, uncertainty, signed prediction error, burden and action expectations adapt from the same public checker outputs. Every trial starts from a complete empty `TrialState`, saved in results. Source/hash identifiers bind snapshots to the immutable trained models and task suite.

This is an **unvalidated transfer** from toy six-knob search to executable code generation. The real model's pretraining may already contain analogous programming problems. Authored tasks are held out from regulator fitting, not proved absent from model pretraining. This is neither a benchmark of new published research mechanisms nor a demonstration of psychological feeling.

## Tasks and independent evaluation

Six software-design families require a pure `solve(data)` implementation: bounded LRU cache state, expiration-correct TTL state, stable resource-constrained batching, accepted-event window deduplication, deterministic dependency scheduling, and bounded rolling statistics. Specifications state ordering, duplicates, capacity/expiry boundaries and storage expectations. All conditions receive the same two public examples and exact requirements; hidden oracle source and cases are withheld from prompts.

Each final candidate has 18 hidden checks: 12 boundary/generated cases, four metamorphic variants, and two scaling workloads. The first five task generators use three parameter values × four lengths (empty, singleton, ordinary and longer generated histories); empty/singleton cases are explicitly labeled boundary, remaining cases generated. Topological scheduling instead has three explicit boundary/cycle cases plus nine generated DAGs. Scaling workloads are separate from these two groups. Metamorphic cases translate times/keys, scale weights, reorder duplicate edges, or translate values while independent oracles recompute the required outputs. Exact recursive output types prevent Python's `True == 1` and `1.0 == 1` from satisfying integer contracts.

The grader receives no controller-condition argument. It computes expected outcomes independently of the proposed code and never accepts the model's claimed success. Correctness is the primary endpoint; wrong but fast code is excluded from correct-only efficiency summaries. Oracles are transparent authored implementations reviewed by executable known-correct/known-incorrect fixtures. They are not independent external human annotation or a verified formal specification.

## Calls, feedback and candidate selection

Default prospective size: six tasks × four generation seeds × two conditions × two candidate calls = **48 trials, at most 96 model calls and 49,152 requested completion tokens**. Input tokens are additional. Each paired condition uses the same model, nonzero sampling temperature 0.6, generation seeds, public information, parsing rules, tool ceiling and allowed operations. Execution order alternates within task/seed pairs to reduce systematic ordering/cache effects. Different generated histories can produce different actual prompt lengths and EOS completion lengths. Exact actual usage and ceilings are recorded; there is no claim of equal consumed tokens.

First generation is followed by exactly **one visible checker call** on two public examples plus one diagnostic case. The feedback discloses those checked inputs, expected and actual outputs, pass/failure status and errors. Wall timings are omitted from the model prompt. No repair call is hidden. Second generation is always the final candidate, even if it fails. Hidden quality never selects a candidate, informs a prompt or updates the controller.

After both generations, final code is independently evaluated on hidden cases. The initial code is then **retrospectively** checked on the same hidden suite, outside the cognitive loop. This gives a valid same-suite revision delta after feedback. Both conditions pay the same one public feedback tool plus two evaluator-only checker calls. Public-first versus hidden-final score difference is retained under an explicitly confounded name; it must not substitute for the same-hidden-suite improvement.

Output is Python source, optionally with a first-line `# confidence: 0.75`. An exact surrounding Python code fence is accepted symmetrically. No extra LLM call repairs malformed output. Parse failures/provider failures consume their model allocation, produce failed checker results, and remain in the denominator. Confidence omission is reported as missing coverage, not filled in with an invented value.

## Execution constraints

Generated code is AST-checked for a defined pure-function subset: no imports, classes, async code, dunder/reflective access, global/nonlocal declarations, dangerous builtins or unlisted methods. The prompt lists the exact allowed builtins and method names. Only plain JSON inputs, minimal builtin functions and generated helper functions are available. Code runs in a fresh isolated-working-directory Python subprocess with address-space, CPU, wall-time, output, open-file, process-count, core-dump and traced-line limits. No filesystem/network API is made available to the candidate.

This is a restricted research execution harness and defense in depth, **not a general OS sandbox for arbitrary adversarial Python**. The user-authorized experiment executes model-generated code constrained to these small pure tasks. Unrestricted repository agents or external unreviewed code require a stronger sandbox. CPU and memory measurements include instrumentation and output allocation; empirical scaling does not prove asymptotic complexity or storage bounds.

## Quality axes and interpretation

The runner records independently:

- Final hidden pass fraction and all-tests-passed rate.
- Boundary, generated-workload and metamorphic robustness.
- Schema/AST validity, execution errors and declared resource-limit compliance.
- Same-hidden-suite improvement after real public feedback.
- Dynamic line events, peak traced allocations, wall time and fourfold input-size line ratios; efficiency eligibility requires correctness.
- Source size, AST node count and a cyclomatic proxy.
- AST fingerprint diversity and public-method operation-signature diversity; these are weak implementation proxies, not mechanism originality or useful creativity.
- Optional probability-of-all-tests-passing confidence, coverage and Brier error against independent hidden correctness.
- Actual prompt/completion tokens, call/tool counts and requested ceilings.

Primary uncertainty is a paired bootstrap of **task means**, keeping generation seeds clustered within tasks. With six task clusters this is exploratory, and multiple secondary metrics are not multiplicity-corrected. Six task families do not establish universal transfer. A positive fixed-schedule or persona comparison is irrelevant to the primary ordinary-adaptive comparison. A null/negative primary result must be retained.

## Reproduce

```bash
python -m unittest discover -s tests -p test_software_eval.py -v
python -m regesh.software_eval \
  --base-url http://127.0.0.1:8080/v1 --model YOUR_SERVED_MODEL \
  --weights results/benchmark --config configs/software_experiment.json \
  --output results/software-real-new --timeout 180
```

The endpoint and model are explicit; no paid credential or model is guessed. Use a new output directory. Artifacts include exact config, model identity, trained-weight hashes, executable source snapshots/hashes, every raw prompt/response/error, controller predictions/scores, within-trial state, tool feedback, generated source, independently checked case outcomes, actual usage, summaries and intervals. Seeds alone do not guarantee deterministic servers; preserved responses supply the audit trail. Model-specific version/quantization/runtime provenance should accompany the run.
