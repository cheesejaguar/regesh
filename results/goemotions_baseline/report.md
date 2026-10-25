# Executed GoEmotions recognition reference

human-label text-emotion recognition only; not functional agent affect or creativity.

Fit 39013 / development 4397 / official test 5427 comments; 28 labels.
Hashed Bernoulli model: 458,976 learned-array bytes; compressed artifact 251,343 bytes.

| Metric | Learned reference | Constant fitting-prior |
|---|---:|---:|
| micro_f1 | 0.414123 | 0.304015 |
| macro_f1 | 0.223103 | 0.017694 |
| micro_precision | 0.377454 | 0.329280 |
| micro_recall | 0.458682 | 0.282351 |
| hamming_loss | 0.054055 | 0.053845 |
| exact_set_accuracy | 0.279160 | 0.295928 |
| mean_sample_jaccard | 0.373018 | 0.312266 |
| brier | 0.034299 | 0.036383 |
| micro_ece | 0.019375 | 0.000579 |
| macro_class_ece | 0.022788 | 0.001926 |
| micro_roc_auc | 0.878885 | 0.770572 |
| macro_roc_auc | 0.780433 | 0.500000 |
| micro_average_precision | 0.347787 | 0.169517 |
| macro_average_precision | 0.197250 | 0.041650 |

Paired comment bootstrap micro-F1 difference 0.110108, 95% interval [0.0969065086464328, 0.12324516960777163]. Unknown author/thread clustering is not modeled.

Raw numerical test predictions/IDs/gold labels are losslessly archived; original Reddit text is omitted. Per-class results and calibration bins are in metrics.json.
All tuning uses development only. A prior baseline receives its own development-selected global threshold. Lower Hamming/Brier/ECE is better; higher F1/accuracy/Jaccard/ranking scores is better.
This small recognition baseline is not an appraiser, persistent emotional controller, psychological exam result, or evidence about useful ideas.
