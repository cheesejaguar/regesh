# Human-label emotion-recognition reference

**Executed 2026-10-25.** This is a small, development-tuned text-recognition reference trained from human GoEmotions annotations. It is **not** the agent's appraisal model, emotional state, persistent controller, creativity mechanism or a human psychology score. The frozen hybrid and ordinary adaptive controllers were not loaded or changed.

The reference makes one downloaded dataset actually usable for supervised training/evaluation without a live model. It also shows why quality needs several measures: a recognizer can improve micro-F1 while becoming worse on exact label sets, rare labels and calibration summaries.

## Reproduce

```bash
python scripts/fetch_datasets.py --cache /workspace/scratch/regesh-datasets --ids goemotions
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m regesh.dataset_baseline \
    --cache /workspace/scratch/regesh-datasets --output results/goemotions-new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_dataset_baseline.py -v
```

The CLI confines its process to one CPU and the executed command limited all common numerical-library thread settings to one. A new/empty output directory is required. No scikit-learn, PyTorch, paid service, model-generated labels or extra dependencies are used.

## Data and frozen protocol

Sources are commit-pinned GoEmotions training/test TSV files and the 28-label vocabulary from the official Google Research repository. Exact source URLs, sizes, SHA-256 hashes and declared Apache-2.0 metadata are in the [manifest](../results/goemotions_baseline/manifest.json). The official card/repository declaration is not an independent guarantee of every upstream Reddit-content right. Raw Reddit text remains outside Git; numerical outputs retain original comment IDs and human label IDs only.

- Official training comments: **43,410**.
- Fitting: **39,013**; development: **4,397**.
- Official held-out test: **5,427**.
- Development membership: first eight hex digits of `SHA256('regesh-goemotions-dev-v1:' + original_ID)`, modulo 10 equals zero. It does not depend on text or labels. Exact split IDs are retained; duplicate or overlapping train/test IDs are rejected.
- The original dataset's separate development file is not used; this experiment derives development only from the downloaded official training partition.

The model is one-vs-rest Bernoulli Naive Bayes with Laplace smoothing alpha=1 and **4,096 hashed binary token-presence features**. Lowercase alphabetic/apostrophe tokens are mapped with CRC32, avoiding process-randomized Python hashes. For each label, it learns positive/negative token-presence rates and a fitting-label prior; it uses absence as well as presence. Neutral and mixed label sets are preserved as annotated.

Only development labels select a temperature from `[.5,1,2,4,8]` by binary-label Brier score, followed by one global multilabel threshold from `.05` to `.80` in `.05` steps by development micro-F1. The chosen temperature is **1.0**, threshold **0.25**. A constant fitting-prior comparator receives its own development-selected threshold, **0.10**. These choices are frozen before official test labels are loaded. No test result was used to choose parameters, tokenization, feature count or thresholds.

Six meaningful checks cover signal learning and safe serialization, tied ranking scores, empty predictions, stable hashing/splits, plain/gzip prediction reading, and a leakage test that reverses held-out labels while requiring identical fitted weights and development-selected settings. A tiny undefined-AUC edge case was repaired before the final archived execution; the full-data numerical results did not change.

## Executed results

| Metric | Learned reference | Constant fitting-prior |
|---|---:|---:|
| Micro-F1 | 0.414123 | 0.304015 |
| Macro-F1 | 0.223103 | 0.017694 |
| Micro precision | 0.377454 | 0.329280 |
| Micro recall | 0.458682 | 0.282351 |
| Exact label-set accuracy | 0.279160 | 0.295928 |
| Mean per-comment Jaccard | 0.373018 | 0.312266 |
| Hamming loss ↓ | 0.054055 | 0.053845 |
| Brier ↓ | 0.034299 | 0.036383 |
| Micro ECE ↓ | 0.019375 | 0.000579 |
| Macro class ECE ↓ | 0.022788 | 0.001926 |
| Micro ROC-AUC | 0.878885 | 0.770572 |
| Macro ROC-AUC | 0.780433 | 0.500000 |
| Micro average precision | 0.347787 | 0.169517 |
| Macro average precision | 0.197250 | 0.041650 |

The reference improves recognition discrimination and micro-F1 over a constant prior. The paired 1,000-resample comment-level bootstrap difference is **+0.110108**, 95% interval **[0.096907, 0.123245]**. These intervals treat comments as independent; unavailable author/thread relationships could make them too narrow.

The result is not uniformly better: exact label sets and Hamming loss are slightly worse, while calibration error summaries are much worse than a constant prior. Predicting population frequencies can be well calibrated while being uninformative about individual comments. Micro ECE also gives substantial weight to easy negative label decisions. Inspect per-class calibration, ranking and recall rather than treating one number as overall quality.

Rare-label recognition is poor. All five classes with fewer than 50 test positives have **F1=0**: embarrassment (37), grief (6), nervousness (23), pride (16), relief (11). The model predicts an average **1.417** labels per comment versus **1.166** human labels and makes an empty prediction on **17.25%** of comments. This is a deliberately modest reference, not a strong modern emotion recognizer.

## Footprint, raw evidence and independent rescoring

The learned weight/bias/prior arrays require **458,976 bytes**; the saved compressed NPZ is **251,343 bytes**. Training and inference timings, peak process RSS and CPU affinity are actual measurements in the manifest, not extrapolated performance claims. The model artifact is loaded with `np.load(..., allow_pickle=False)` and validated shapes/numbers.

`results/goemotions_baseline/` retains the exact implementation snapshot, split IDs, model, tuning grid/results, environment/data/source hashes, all per-class precision/recall/F1/ranking/calibration scores, calibration bins and losslessly compressed test predictions. `verification.json` confirms every final metric can be recomputed from the numerical archive without reading original Reddit text, and that the final micro-F1 matches the initial execution. File hashes in the manifest identify the exact artifacts.

```python
from regesh.dataset_baseline import BernoulliReference, read_predictions
model = BernoulliReference.load('results/goemotions_baseline/reference.npz')
rows = read_predictions('results/goemotions_baseline/predictions.jsonl.gz')
first = next(rows)  # ID, human/predicted label IDs, probabilities; no source text
```

Bernoulli conditional independence, hashing collisions, tokenization, class imbalance and one global threshold constrain performance and probability quality. Human emotion labels can disagree, and text-perceived emotion is not direct evidence about a speaker's internal condition. This supervised recognition result establishes neither emotional experience nor better agent problem solving. The independently run real-model software and psychology comparisons answer different questions.
