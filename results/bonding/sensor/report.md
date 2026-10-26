# Partner-state sensor: affect evidence from the reply model's own embeddings

Protocol frozen 2026-10-26T03:00:00Z; run started 2026-10-26T05:37:47.208117+00:00. Numbers are single frozen-run test results with 95% item-bootstrap intervals.

## E1 GoEmotions (fit 13599 / development 1494 / test 5427; dimension 1536)

| Model | Threshold | Micro-F1 [95%] | Macro-F1 [95%] | Support-weighted F1 | Hamming | Brier | Micro ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear | 0.25 | 0.4856 [0.4760, 0.4951] | 0.3483 [0.3242, 0.3684] | 0.4716 | 0.0475 | 0.0293 | 0.0190 |
| mlp | 0.30 | 0.4743 [0.4636, 0.4845] | 0.3006 [0.2873, 0.3121] | 0.4513 | 0.0463 | 0.0302 | 0.0097 |
| bernoulli_35 | 0.25 | 0.3697 [0.3591, 0.3804] | 0.1406 [0.1345, 0.1469] | 0.3166 | 0.0611 | 0.0355 | 0.0170 |
| constant_prior | 0.10 | 0.3040 [0.2928, 0.3156] | 0.0177 [0.0172, 0.0182] | 0.1399 | 0.0538 | 0.0364 | 0.0007 |
| archived reference (100% train) | 0.25 | 0.4141 | 0.2231 | 0.3939 | | | |

Paired differences (micro-F1, macro-F1):

- linear-bernoulli_35: micro +0.1159 [+0.1055, +0.1266]; macro +0.2077 [+0.1835, +0.2301]
- linear-constant_prior: micro +0.1816 [+0.1694, +0.1945]; macro +0.3306 [+0.3063, +0.3510]
- mlp-bernoulli_35: micro +0.1046 [+0.0940, +0.1159]; macro +0.1601 [+0.1455, +0.1730]
- mlp-constant_prior: micro +0.1702 [+0.1580, +0.1822]; macro +0.2830 [+0.2696, +0.2945]
- mlp-linear: micro -0.0113 [-0.0196, -0.0030]; macro -0.0476 [-0.0694, -0.0255]

Deployed sensor: `linear` head, digest `c78e4218de10948b1e478963e1811b150b634ae06addcb5c15c6a93098118a3a`.

## E2 DailyDialog (train 17568 turns from 2258 dialogues / valid 8069 / test 7740)

| Model | Accuracy [95%] | Macro-F1 [95%] | Valence MAE [95%] |
|---|---:|---:|---:|
| linear | 0.8426 [0.8345, 0.8509] | 0.3069 [0.2851, 0.3270] | 0.1463 [0.1381, 0.1545] |
| mlp | 0.8404 [0.8328, 0.8486] | 0.3069 [0.2860, 0.3267] | 0.1483 [0.1404, 0.1558] |
| bernoulli_20 | 0.8193 [0.8106, 0.8277] | 0.1869 [0.1824, 0.1917] | 0.1686 [0.1605, 0.1768] |
| majority | 0.8167 [0.8083, 0.8245] | 0.1284 [0.1277, 0.1291] | 0.1683 [0.1605, 0.1764] |

Zero-shot GoEmotions sensor on DailyDialog test turns: sign agreement 0.8941 [0.8769, 0.9109] on 1303 emotional turns (majority sign 0.7820, chance .5); ROC-AUC positive vs negative 0.9337965970504084; three-way accuracy with dead band 0.1: 0.4484 vs 0.8317 for predicting 0 everywhere.

Recognition quality of a frozen-embedding head against hashed-token references on labelled corpora. The 35% / 20% training subsets, the fixed valence map and the proxy nature of support are declared in config.json. Nothing here measures benefit to users; sensor outputs are machine interpretations that fill affect evidence only and never establish facts, authority or permissions.
