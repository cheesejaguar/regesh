# Partner-state sensor: affect evidence from the reply model's own embeddings

Date: 2026-10-26. Protocol frozen `2026-10-26T03:00:00Z` in
`configs/bonding_sensor.json`; single frozen run `2026-10-26T05:37Z`, outputs,
heads, numeric predictions and SHA-256 manifest in `results/bonding/sensor/`.
This report describes what the sensor is, how it was measured on two labelled
corpora against the repository's existing hashed-token references, and what it
does not establish. Recognition quality on labelled text is not benefit to
users; the sensor's outputs are machine interpretations of the user's words.

## Why a sensor, and what it is not

The bond mechanisms (`docs/bonding.md`) consume two evidence fields on each
partner Event: `partner_affect` in [-1, 1] drives the online dyadic identifier
and the steering recommendation; `support` in [0, 1] is comfort for arousal
regulation and the answer to a pending support request. Until now a host had to
supply both by hand or from a simulator. The sensor estimates them from the
user's text with a small head trained on **mean-pooled embeddings of the same
local language model that writes the agent's replies** (Qwen2.5-Coder-1.5B-
Instruct, Q4_K_M, served by llama.cpp with `--embedding --pooling mean`). The
language model is not fine-tuned; the head is the only trained component.

Boundary: the sensor is an optional evidence provider the **host** runs before
`agent.observe`. The Agent never calls the network, and `regesh.bonding.sensor`
never does either; the host obtains the embedding (through
`regesh.bonding.embeddings.EmbeddingClient`, or a cache) and passes the vector
in. Everything the sensor adds is recorded under `data["sensor"]` (model digest,
the 28 probabilities, the valence map name, what was filled), so machine-estimated
evidence stays distinguishable from hand-supplied evidence in the journal and in
replay. Sensor outputs fill affect evidence only; they never establish facts,
authority or tool permissions (the trust firewall is unchanged).

## Declared maps

**Valence** (`goemotions-sentiment-v1`): the GoEmotions paper's sentiment
grouping (Demszky et al. 2020). Positive (+1): admiration, amusement, approval,
caring, desire, excitement, gratitude, joy, love, optimism, pride, relief.
Negative (-1): anger, annoyance, disappointment, disapproval, disgust,
embarrassment, fear, grief, nervousness, remorse, sadness. Ambiguous (0):
confusion, curiosity, realization, surprise. Neutral (0). The value is the
mass-weighted mean sign over all 28 sigmoid probabilities,
`sum(sign_l p_l) / sum(p_l)`, clipped to [-1, 1] (0 when there is no mass).
Neutral and ambiguous mass therefore pulls the estimate toward 0.

**Support proxy** (`goemotions-support-noisy-or-v1`): `1 - prod(1 - p_l)` over
caring, gratitude, approval, love and admiration: the probability that at least
one supportive label is present. It is a **proxy**: it detects supportive
language in the user's text, not that the user answered the agent's request. It
is off by default; `annotate_event(..., fill_support=True)` opts in.

## Wiring the sensor into a host

```python
from regesh import Agent, Event
from regesh.bonding.embeddings import EmbeddingClient
from regesh.bonding.sensor import Sensor, annotate_event

sensor = Sensor.load("results/bonding/sensor/sensor.json")      # head + standardizer + maps, digest-checked
client = EmbeddingClient("http://127.0.0.1:8080/v1", "regesh-qwen-coder")   # host side only
agent = Agent("agent.sqlite", "companion", config={"bonding": True, "partner": "user"})

text = "thanks, that actually helped"
event = Event("u-17", "chat", "user", "agent", "conversation", "2026-10-26T06:00:00Z", data={"text": text})
embedding = client.embed([text])[0]                              # 1536-d, from the reply model's server
agent.observe(annotate_event(event, sensor, embedding))          # fills partner_affect, records data["sensor"]
# agent.observe(annotate_event(event, sensor, embedding, fill_support=True))   # opt-in support proxy
```

Rules: `annotate_event` returns a new Event and never overwrites a field the host
supplied; `partner_affect` is filled when absent; `support` only when absent and
`fill_support=True`; `data["sensor"]` is always written. The llama-server must be
started with `--embedding --pooling mean` (the configuration the cache job used);
`sensor.estimate(embedding)` alone returns `{"emotion_probabilities", "partner_affect",
"support_proxy"}` for hosts that want the numbers without an Event. The sensor
checks the embedding dimension against its standardizer and refuses a mismatch,
and `Sensor.load` refuses a bundle whose digest does not match its contents.

## Protocol (frozen before any training on real vectors)

Embeddings: 1536-d mean-pooled final hidden states of the pinned model, served by
llama.cpp (`--embedding --pooling mean`), cached as float16 keyed by the SHA-256
of the exact text by a separate job (`regesh.bonding.embeddings`); the trainer
never queries the server (`live_model_calls: 0`). Cached vectors are unit-norm,
finite, and strongly anisotropic (random-pair cosine .73-.75), so every head
works on per-dimension standardized vectors with mean and standard deviation
taken from the training rows only (smallest training standard deviation
5.8e-5, about 2,300 float16 steps; quantization is not a limiting factor).

Heads: a linear logistic head `[1536, k]` and a one-hidden-layer MLP
`[1536, 256, k]` (tanh), both sigmoid binary cross-entropy, Adam (lr 1e-3, l2
1e-4, batch 128, at most 60 epochs, patience 6, seed 0), early stopping on
development/validation BCE with best-epoch weights restored. No hyperparameter
search; one seed. Bootstrap: 1,000 item resamples, seed 20261026, the same
resamples for every model so differences are paired.

**E1, GoEmotions (28 labels, multilabel).** Training rows are the hash-selected
35% of the official train split (`select_by_hash(ids, .35, "goemotions-train")`,
15,093 rows), minus the development rows of the archived baseline's ID-hash rule
(1,494), leaving 13,599 fit rows. Development rows choose the single global
threshold on the baseline's grid (.05-.80 by .05) by micro-F1; the heads get no
temperature scaling. The official test split (5,427 rows) is scored once per
model with `regesh.dataset_baseline.evaluate`. Comparators: a constant prior
from the same fit rows with its own development threshold; the hashed-token
Bernoulli reference (`BernoulliReference`) refit on exactly the same 13,599 rows
and calibrated on the same 1,494 (like for like); and the archived reference
(`results/goemotions_baseline/`, 39,013 fit rows, 100% of train), quoted.
The deployed sensor is the head with the higher development micro-F1.

**E2, DailyDialog (7 turn emotions).** Training turns come from the
hash-selected 20% of train dialogues (`select_by_hash(["train-i"], .2,
"dailydialog-train")`: 2,258 dialogues, 17,568 turns, 82% no_emotion); all 8,069
valid turns early-stop; all 7,740 test turns are scored once. Sigmoid BCE on
one-hot targets with argmax decoding (the shared MLP has no softmax).
Baselines: majority class and the Bernoulli reference fit on the same turns with
argmax decoding. Valence scale: happiness +1; anger, disgust, fear, sadness -1;
no_emotion and surprise 0. The deployed GoEmotions sensor is also applied
unchanged (its own standardizer) to the DailyDialog test turns, and its valence
sign is compared with the gold sign on the 1,303 turns whose gold sign is
non-zero, against chance (.5) and the majority sign (+1 everywhere, which scores
the positive fraction, .782).

Pilots declared in the config: unit tests on synthetic vectors, cache coverage
and vector-statistics checks on the test caches, and one train/evaluate pilot on
two hash-halves of the DailyDialog **valid** split (no test data) to confirm
runtime and a learning signal. Nothing was changed after the freeze.

## Results: E1 GoEmotions, official test set (5,427 rows)

| Model | Fit rows | Threshold | Micro-F1 [95%] | Macro-F1 [95%] | Support-weighted F1 | Hamming ↓ | Brier ↓ | Micro ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Linear head (deployed)** | 13,599 | .25 | **.4856** [.4760, .4951] | **.3483** [.3242, .3684] | .4716 | .0475 | .0293 | .0190 |
| MLP head | 13,599 | .30 | .4743 [.4636, .4845] | .3006 [.2873, .3121] | .4513 | .0463 | .0302 | .0097 |
| Bernoulli reference, same 35% | 13,599 | .25 (T 2.0) | .3697 [.3591, .3804] | .1406 [.1345, .1469] | .3166 | .0611 | .0355 | .0170 |
| Constant prior, same 35% | 13,599 | .10 | .3040 [.2928, .3156] | .0177 [.0172, .0182] | .1399 | .0538 | .0364 | .0007 |
| Archived Bernoulli reference (100%) | 39,013 | .25 | .4141 | .2231 | .3939 | .0541 | .0343 | .0194 |

Paired item-bootstrap differences (micro-F1; macro-F1):

| Contrast | Micro-F1 | Macro-F1 |
|---|---:|---:|
| Linear - Bernoulli 35% (like for like) | **+.1159** [+.1055, +.1266] | **+.2077** [+.1835, +.2301] |
| Linear - constant prior | +.1816 [+.1694, +.1945] | +.3306 [+.3063, +.3510] |
| MLP - Bernoulli 35% | +.1046 [+.0940, +.1159] | +.1601 [+.1455, +.1730] |
| MLP - linear | -.0113 [-.0196, -.0030] | -.0476 [-.0694, -.0255] |
| Linear - archived reference (post hoc, see below) | +.0716 [+.0607, +.0823] | +.1252 [+.1002, +.1486] |

Reading. With 35% of the training rows, a logistic head on the reply model's
frozen embeddings beats the hashed-token reference trained on the same rows by
.12 micro-F1 and .21 macro-F1, and also exceeds the archived reference that saw
2.9 times more rows (.486 vs .414 micro-F1; .348 vs .223 macro-F1). The
training-size handicap therefore runs against the heads, not for them. The
contrast with the archived reference was not part of the frozen protocol
(the config quotes the archived number); the paired interval in the last row is
a post hoc check computed from the two archived per-ID prediction files
(`results/goemotions_baseline/predictions.jsonl.gz` and
`results/bonding/sensor/goemotions_predictions.jsonl.gz`, same bootstrap seed),
and the archived micro-F1 recomputed from its own predictions is .414123, as in
its `metrics.json`. The MLP is reliably a little worse than the linear head on
F1 despite a lower best development BCE (.1127 vs .1152): it stopped at epoch 13
and over-fits the 13.6k rows, while the linear head used all 60 epochs with the
development loss still falling (.681 to .115), so the linear head is **not
converged under the frozen budget**; early stopping never fired for it. The MLP
is better calibrated (micro ECE .0097 vs .0190; the linear head's ECE equals the
archived reference's), predicts fewer labels (1.30 vs 1.42 per comment; gold
1.17) and leaves 15.7% of comments without a label (linear 9.6%, archived
17.2%).

Per class, the linear head's largest gains over the 35% Bernoulli reference are
remorse (.52 vs .00), fear (.48 vs .00), curiosity (.47 vs .13), desire (.33 vs
.00) and surprise (.36 vs .03); the archived reference had F1 0 on all five
labels with fewer than 50 test positives, the linear head scores grief .29,
relief .17 and nervousness .14 but still 0 on embarrassment (37) and pride (16).
Full per-class precision/recall/F1/Brier/ECE/AUC rows for every model are in
`results/bonding/sensor/goemotions_metrics.json`.

## Results: E2 DailyDialog, full test split (7,740 turns)

| Model | Accuracy [95%] | Macro-F1 [95%] | Valence MAE ↓ [95%] |
|---|---:|---:|---:|
| Linear head | **.8426** [.8345, .8509] | **.3069** [.2851, .3270] | **.1463** [.1381, .1545] |
| MLP head | .8404 [.8328, .8486] | .3069 [.2860, .3267] | .1483 [.1404, .1558] |
| Bernoulli reference, same 20% (argmax) | .8193 [.8106, .8277] | .1869 [.1824, .1917] | .1686 [.1605, .1768] |
| Majority (no_emotion) | .8167 [.8083, .8245] | .1284 [.1277, .1291] | .1683 [.1605, .1764] |

Paired differences, linear head minus Bernoulli: accuracy +.0234 [+.0165,
+.0305], macro-F1 +.1200 [+.0977, +.1408], valence MAE -.0224 [-.0291, -.0156];
minus majority: +.0260, +.1785, -.0221 (all intervals exclude zero). The heads
recognise happiness (F1 .51/.53) and surprise (.46/.47), weakly anger (.13/.12)
and sadness (.14/.13), and never predict disgust or fear (47 and 17 test turns;
77 and 31 training turns). The Bernoulli reference finds only no_emotion and
happiness. The probability-weighted "soft" valence is worse than the argmax
valence (MAE .196 / .187), because the sigmoid outputs do not sum to one and
the no_emotion mass dilutes the emotional classes. The linear head again ran
all 60 epochs; the MLP stopped at epoch 9.

**Zero-shot transfer of the GoEmotions sensor** (no DailyDialog training), on
the 1,303 test turns with a non-zero gold sign: sign agreement **.894** [.877,
.911] against .782 for always predicting positive and .5 for chance; ROC-AUC of
the sensor valence for positive versus negative turns .934 (average precision
.979). Mean sensor valence by gold emotion: happiness +.43 (n 1,019), surprise
+.03 (116), fear -.06 (17), anger -.14 (118), sadness -.19 (102), disgust -.21
(47), **no_emotion +.12 (6,321)**. The last number is the important caveat: on
neutral conversational turns the sensor's mean absolute valence is .20 (emotional
turns .40) with a positive bias, so the declared three-way check with a +-.1
dead band scores only .448 against .832 for predicting 0 everywhere (3,097 of
6,321 no_emotion turns land above +.1). The mass-weighted sign map does not
return near-zero values for small talk; a host that feeds every turn into the
dyadic identifier will see a mildly positive partner most of the time. The mean
support proxy on DailyDialog test turns is .31; DailyDialog has no support label,
so the proxy is **unvalidated** here.

## Deployed artefact

`results/bonding/sensor/sensor.json` (1.0 MB): the linear GoEmotions head
(43,036 parameters, digest in `summary.json`), its standardizer (1536 means and
standard deviations from the 13,599 fit rows), threshold .25, the valence map
and the support-proxy definition, and a bundle digest that `Sensor.load`
verifies. The heads are also saved alone via `MLP.save`
(`goemotions_linear.json`, `goemotions_mlp.json`, `dailydialog_linear.json`,
`dailydialog_mlp.json`), the two Bernoulli comparators as NPZ, numeric
predictions per test item without any text (`*_predictions.jsonl.gz`), the
exact fit/development IDs and selected dialogue IDs (`split_ids.json`), the
config and source copies, `summary.json` (histories, all metrics, bootstrap
blocks, embedding manifests with their `selected_ids_sha256`) and
`manifest.json` (SHA-256 of every file, the cache files, the config and the
five source modules). Total 21 MB, of which 17.6 MB are the two MLP heads in
full-precision JSON; the whole run took 135 s on two BLAS threads of a shared
four-core CPU (head training 15 s / 22 s for E1, 11 s / 22 s for E2).

## Limitations

- **Representation.** The vectors are mean-pooled hidden states of a 1.5B
  decoder-only *code* instruct model, not a sentence-embedding model, and the
  language model is not fine-tuned. The head can only use what that
  representation already separates; the anisotropy (random-pair cosine .73-.75)
  is removed only by standardization.
- **Training subsets.** 35% of GoEmotions train and 20% of DailyDialog train
  dialogues were embedded on the shared CPU server. The GoEmotions handicap runs
  against the heads (they still win); the DailyDialog minority classes (fear 31,
  disgust 77 training turns) are too small for the heads to learn.
- **Not converged.** The linear heads used the full 60-epoch budget with the
  development loss still falling; the reported linear numbers are a lower bound
  on what the same head reaches with more epochs. The MLP over-fits and stops
  early. Neither was tuned, by design.
- **Domain shift.** GoEmotions is Reddit comments with multilabel annotation;
  DailyDialog is scripted everyday dialogue, 83% no_emotion, one label per
  turn. The sensor will meet the same shift with real users, and E2 shows what
  it costs: good sign agreement when the user is emotional, a positive bias and
  non-zero valence when the user is not.
- **Valence map.** A fixed three-valued sign map flattens intensity, treats
  ambiguous emotions and neutral as 0, inherits GoEmotions' annotation biases,
  and (as measured) does not produce near-zero outputs on neutral text.
- **Support proxy.** It detects supportive language; it is not evidence that the
  user answered the agent's request, it was not validated on any labelled data
  here, and it stays off unless the host opts in.
- **Statistics.** Item bootstraps ignore author/thread (GoEmotions) and dialogue
  (DailyDialog) clustering and may be too narrow; one seed per head; the
  archived-reference interval is post hoc.
- **Scope.** Sensor outputs fill affect evidence only; they never establish
  facts, authority or tool permissions. No human validation of the mapped
  valence against users' own reports was run; recognition on labelled corpora is
  not benefit to users.

## Claim ledger entries

| Label | Scope |
|---|---|
| Established precedent | Frozen-embedding probes for emotion recognition; the GoEmotions taxonomy and sentiment grouping; DailyDialog emotion labels. None is claimed as new. |
| Engineering synthesis | One digest-checked sensor bundle (head, standardizer, declared maps) that fills `partner_affect` and, opt-in, `support` on an Event with explicit provenance, using the reply model's own embeddings so no second model is needed, and kept outside the Agent's trust boundary. |
| Measured result | On the official GoEmotions test set a logistic head on 35% of the training rows reaches micro-F1 .486 [.476, .495] and macro-F1 .348 [.324, .368], +.116 / +.208 over the hashed-token reference trained on the same rows and +.072 / +.125 (post hoc) over the archived 100% reference. On DailyDialog the heads beat majority and Bernoulli baselines on accuracy, macro-F1 and valence MAE, and the GoEmotions sensor's zero-shot valence sign agrees with 89.4% [87.7, 91.1] of emotional turns (majority sign 78.2%, AUC .934) while returning a mean +.12 valence on neutral turns. |
| Speculative hypothesis | That sensor-filled evidence improves the bond's dyadic identification or steering with real users. Untested; the positive bias on neutral turns is a known hazard for the identifier. |

What would change the ledger: a longer or tuned training budget for the linear
head; a neutral-aware map (for example, a declared dead band chosen on
development data, or subtracting the no_emotion-turn mean measured on a
development conversation set); a labelled conversational support signal to
validate the proxy; and a user study comparing hand-supplied, sensor-filled and
no partner-affect evidence on the bond's measured outcomes.

## Reproduce

```bash
# Populate the caches once (separate job; the trainer never queries the server):
python -m regesh.bonding.embeddings --dataset goemotions --split test --cache .cache/regesh-data --out .cache/embeddings --base-url http://127.0.0.1:8080/v1 --model regesh-qwen-coder
python -m regesh.bonding.embeddings --dataset goemotions --split train --fraction 0.35 --cache .cache/regesh-data --out .cache/embeddings --base-url http://127.0.0.1:8080/v1 --model regesh-qwen-coder
python -m regesh.bonding.embeddings --dataset dailydialog --split test  --cache .cache/dailydialog/parlai --out .cache/embeddings --base-url http://127.0.0.1:8080/v1 --model regesh-qwen-coder
python -m regesh.bonding.embeddings --dataset dailydialog --split valid --cache .cache/dailydialog/parlai --out .cache/embeddings --base-url http://127.0.0.1:8080/v1 --model regesh-qwen-coder
python -m regesh.bonding.embeddings --dataset dailydialog --split train --fraction 0.2 --cache .cache/dailydialog/parlai --out .cache/embeddings --base-url http://127.0.0.1:8080/v1 --model regesh-qwen-coder
# Frozen run (empty output directory required):
OPENBLAS_NUM_THREADS=2 python -m regesh.bonding.train_sensor --config configs/bonding_sensor.json \
  --embeddings .cache/embeddings --data .cache/regesh-data --dailydialog .cache/dailydialog/parlai \
  --output results-new/bonding-sensor
python -m unittest tests.test_bonding_sensor -v
```
