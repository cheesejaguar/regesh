# Echoes of power: reproducing the Supreme Court coordination results

Execution date: 2026-10-25. Primary source: Danescu-Niculescu-Mizil, Lee, Pang and
Kleinberg (2012), [Echoes of power: Language effects and power differences in social
interaction](https://arxiv.org/abs/1112.3670), WWW 2012, peer reviewed. This is an
**independent implementation on public data**: the measure is the existing
`regesh.bonding.synchrony` code (Eq. 1-3 and Aggregated 1-3 of the paper), the data are
the ConvoKit `supreme-corpus` term files 2004-2006, and the case set is the one the
ConvoKit coordination example notebook lists as the cases analysed in the original paper.
It is not a rerun of the authors' code, transcripts or LIWC dictionary. The notebook
itself warns that its code and data are not identical with the ones used in the paper.

```bash
python -m regesh.reproductions.coordination_power \
  --cache .cache/convokit/supreme --output results-new/coordination-power
```

The output directory must be absent or empty. The runner reads nothing from the network;
every input must match the SHA-256 recorded in `configs/reproductions/coordination_power.json`,
which also freezes the 204 case ids, the alias map, the thresholds, the statistics and the
declared ambiguities. The results in `results/papers/coordination_power/` contain the config
copy, the runner copy, `summary.json`, `manifest.json` with artifact hashes, a machine-written
`report.md`, and `entities.jsonl.gz` with per-entity sufficient statistics (marker counts
`a`, `b`, `ab` and exchange count `n`) and coordination values. No utterance text is written.
The whole run takes about 18 s on one CPU.

## The lexicon caveat, stated first

The paper used eight LIWC-derived function-word categories (451 lexemes). `synchrony.LEXICON`
is an authored open list (56 personal pronouns, 37 impersonal pronouns, 3 articles, 34
conjunctions, 65 prepositions, 76 auxiliary verbs, 68 adverbs, 49 quantifiers). Because
C^m depends on how often each category fires, **absolute coordination levels are not
comparable with Figures 4-5**, and neither are the per-marker entity counts, which depend on
how often a category is exhibited at all. What can be compared is the sign of each contrast
and whether it is statistically distinguishable from zero. All values below are on the
paper's percent scale (x100).

## Data and coverage

| Quantity | Published (Section 3.2) | This run |
|---|---:|---:|
| Argument sessions / cases | 204 cases | 204 sessions, 201 ConvoKit case ids |
| Verbal exchanges | 50,389 | 49,281 Justice-lawyer exchanges (+1,316 Justice-to-Justice and 457 unknown-speaker replies excluded) |
| Justices | 11 (Thomas, Alito sparse) | 11, same names |
| Lawyers | 311 | 312 distinct ids; 524 (lawyer, case) entities; 1,519 (Justice, case) entities |
| Lawyers in one case only | 73% | 80.4% |
| Maximum cases per lawyer | 15 | 22 |

The notebook's `case_ids` set (SHA-256 `9783bbef...d3d2`, fetched 2026-10-25 from the
ConvoKit master branch, HEAD `91a1df01...8fa`; the per-file commit was not obtainable
because GitHub API access is disabled in this session) has 204 entries. 199 match ConvoKit
case ids literally. The other five are id variants: `128orig` and `105original` are the two
original-jurisdiction cases of the 2004 term, stored as `128-orig` and `105-orig`; `04-1170b`,
`04-1360b` and `04-473b` are second argument sessions of cases already in the set, which
ConvoKit stores under the same case id. With the declared alias map every one of the 204
sessions is covered (68/78/58 per term). The literal notebook filter, which silently drops
the two original-jurisdiction cases (199 case ids, 202 sessions), is run as a sensitivity.
Case metadata are internally consistent: `votes_side` and advocate sides in
`conversations.json` agree with `cases.jsonl` for all 204 sessions; the SCDB table in the
cache is checksummed but not needed.

Exchanges are immediate replies (`reply_to`) within a session between speakers of
different type (J, A), with non-empty text on both sides (no empty texts occur). Entities are
(speaker, case) per footnote 14. Amici and unknown-side advocates count as lawyers for the
status analyses, as in Section 3.2, but have no defined favorability: of 24,638 lawyer
replies to Justices, 13,731 are to an unfavorable Justice, 9,802 to a favorable one, 954 come
from amici or unknown-side advocates and 156 target a Justice without a recorded 0/1 vote;
the Justice-reply split is 13,765 / 9,769 / 953 / 151. Rehnquist appears only in the 2004
term file and Roberts only in 2005-2006, as expected for the Chief Justice contrast.

## Thresholds and statistics

The paper does not state minimum-data rules. The primary analysis adopts ConvoKit's rule
with the values of its Wikipedia example (`speaker_thresh=7`, `target_thresh=7`): C^m(b, A)
is defined only when at least seven target utterances in the exchange set exhibit m and at
least seven of b's replies exhibit m. These are counts of marker-exhibiting utterances, not
plain utterance counts, which is what the ConvoKit code does
(`Coordination._scores_over_utterances`, SHA-256 `2a117527...0909`). Two threshold
sensitivities are run on the same counts: the notebook's Supreme Court values (3/3) and plain
Eq. (1) with at least seven exchanges per entity. Group values come from
`synchrony.group_coordination`; per-entity Aggregated 1-3 values are rebuilt with the same
rules and asserted to average to the library's group values.

Tests compare the hypothesised higher group with the lower group across entities: Welch's
two-sided t-test (p-values from `scipy.stats.t`; a normal approximation is the fallback),
Student's t as reported by the paper's caption, a percentile bootstrap interval of the
difference of group means (10,000 resamples within each group, seed 20261025), and, where the
same entity appears in both groups, a paired t-test as a sensitivity. No multiplicity
correction is applied, matching the paper's uncorrected stars; the number of significant
markers is reported descriptively. The 7/7 rule is strict for the smaller exchange sets:
only 235 of 1,517 Justice entities have all eight markers defined (the paper had 815 of
1,307), and only 12 lawyer entities have all eight defined toward the Chief Justice, so
Aggregated 1 is underpowered here and Aggregated 2 and 3 carry the evidence. Under the 3/3
sensitivity, 501 lawyer and 912 Justice entities are complete, close to the paper's 500 and 815,
which suggests the paper's effective rule was looser than 7/7 for the categories as fired by this
lexicon; the 7/7 primary was fixed before any result was seen and is kept as declared.

## Results

Each table lists entity counts for the two groups, group means (percent), the difference
in the hypothesised direction, Welch t and p, the bootstrap interval and the paper's star
for that quantity (* p<.05, ** p<.01, *** p<.001; blank means not significant in the paper;
the Chief Justice contrast has only "p<0.01" in the text).

### F4_lawyers_coordinate_more_than_justices

Higher (hypothesis): `lawyers_to_justices`; lower: `justices_to_lawyers`. Entities: {'lawyers_to_justices': 524, 'justices_to_lawyers': 1517}.

| Quantity | n (high, low) | mean high % | mean low % | diff % | Welch t | p | bootstrap 95% CI % | published |
|---|---|---:|---:|---:|---:|---:|---|---|
| aggregated_1 | [415, 235] | 3.126 | 2.453 | 0.673 | 2.33 | 0.0203 | [0.114, 1.235] | *** |
| aggregated_2 | [519, 1181] | 2.978 | 2.213 | 0.765 | 4.37 | 0.0000 | [0.418, 1.114] | *** |
| aggregated_3 | [519, 1181] | 2.834 | 1.658 | 1.176 | 5.77 | 0.0000 | [0.772, 1.571] | *** |
| ppron | [515, 997] | 3.689 | 1.787 | 1.902 | 6.14 | 0.0000 | [1.301, 2.511] | *** |
| ipron | [516, 1074] | 2.253 | 1.237 | 1.017 | 3.57 | 0.0004 | [0.455, 1.583] | *** |
| article | [516, 1036] | 1.921 | 1.957 | -0.036 | -0.11 | 0.9126 | [-0.695, 0.597] | * |
| conj | [513, 958] | 1.871 | 2.540 | -0.668 | -1.85 | 0.0647 | [-1.381, 0.054] |  |
| prep | [515, 1038] | 2.004 | 1.416 | 0.588 | 1.88 | 0.0605 | [-0.029, 1.189] | *** |
| auxverb | [518, 1146] | 1.718 | 0.850 | 0.868 | 3.96 | 0.0001 | [0.450, 1.299] | *** |
| adverb | [501, 700] | 4.344 | 2.890 | 1.454 | 2.56 | 0.0106 | [0.325, 2.571] | * |
| quant | [418, 246] | 6.022 | 5.026 | 0.996 | 1.00 | 0.3181 | [-0.933, 3.006] | ** |

### fine_status_chief_over_associates

Higher (hypothesis): `lawyers_to_chief`; lower: `lawyers_to_associates`. Entities: {'lawyers_to_chief': 393, 'lawyers_to_associates': 523}.

| Quantity | n (high, low) | mean high % | mean low % | diff % | Welch t | p | bootstrap 95% CI % | published |
|---|---|---:|---:|---:|---:|---:|---|---|
| aggregated_1 | [12, 371] | 4.160 | 3.147 | 1.013 | 0.93 | 0.3729 | [-0.919, 3.156] |  |
| aggregated_2 | [145, 516] | 3.307 | 3.082 | 0.225 | 0.67 | 0.5067 | [-0.416, 0.904] |  |
| aggregated_3 | [145, 516] | 2.535 | 2.886 | -0.351 | -0.68 | 0.4984 | [-1.336, 0.697] |  |
| ppron | [117, 504] | 3.018 | 3.647 | -0.629 | -0.90 | 0.3693 | [-1.937, 0.779] |  |
| ipron | [102, 509] | 2.162 | 2.388 | -0.225 | -0.31 | 0.7578 | [-1.649, 1.247] |  |
| article | [104, 509] | 2.021 | 2.101 | -0.080 | -0.08 | 0.9379 | [-2.003, 1.975] |  |
| conj | [86, 501] | 2.396 | 2.293 | 0.103 | 0.12 | 0.9062 | [-1.581, 1.858] |  |
| prep | [115, 511] | 1.890 | 1.923 | -0.033 | -0.04 | 0.9661 | [-1.525, 1.474] |  |
| auxverb | [136, 512] | 2.679 | 1.641 | 1.038 | 1.90 | 0.0590 | [-0.007, 2.137] |  |
| adverb | [43, 476] | 2.602 | 4.663 | -2.062 | -1.05 | 0.2997 | [-5.877, 1.722] |  |
| quant | [12, 375] | 9.685 | 6.000 | 3.685 | 1.45 | 0.1707 | [-0.837, 8.645] |  |

Paired sensitivity (entities present in both groups): aggregated_1: n=7, diff=2.591%, p=0.0518; aggregated_2: n=145, diff=-0.149%, p=0.6836; aggregated_3: n=145, diff=-0.808%, p=0.1035

### F5a_lawyers_to_unfavorable_over_favorable

Higher (hypothesis): `lawyers_to_unfavorable`; lower: `lawyers_to_favorable`. Entities: {'lawyers_to_unfavorable': 363, 'lawyers_to_favorable': 377}.

| Quantity | n (high, low) | mean high % | mean low % | diff % | Welch t | p | bootstrap 95% CI % | published |
|---|---|---:|---:|---:|---:|---:|---|---|
| aggregated_1 | [235, 161] | 3.355 | 2.556 | 0.799 | 2.27 | 0.0236 | [0.112, 1.483] | * |
| aggregated_2 | [335, 312] | 3.365 | 2.385 | 0.981 | 3.71 | 0.0002 | [0.472, 1.492] | *** |
| aggregated_3 | [335, 312] | 3.108 | 2.124 | 0.984 | 3.11 | 0.0019 | [0.367, 1.596] | *** |
| ppron | [322, 287] | 3.804 | 3.167 | 0.637 | 1.40 | 0.1631 | [-0.251, 1.542] | ** |
| ipron | [326, 299] | 2.675 | 2.191 | 0.483 | 1.09 | 0.2767 | [-0.398, 1.349] | * |
| article | [325, 285] | 2.654 | 1.052 | 1.601 | 3.18 | 0.0016 | [0.628, 2.597] | * |
| conj | [317, 274] | 2.480 | 1.553 | 0.926 | 1.74 | 0.0822 | [-0.089, 1.968] |  |
| prep | [326, 283] | 2.354 | 1.476 | 0.879 | 1.92 | 0.0556 | [-0.033, 1.748] | ** |
| auxverb | [334, 307] | 1.725 | 1.580 | 0.145 | 0.43 | 0.6667 | [-0.501, 0.791] |  |
| adverb | [294, 251] | 4.471 | 3.713 | 0.758 | 0.95 | 0.3428 | [-0.823, 2.330] | * |
| quant | [237, 161] | 6.761 | 4.346 | 2.415 | 2.05 | 0.0412 | [0.126, 4.695] | * |

Paired sensitivity (entities present in both groups): aggregated_1: n=34, diff=1.461%, p=0.0499; aggregated_2: n=169, diff=1.104%, p=0.0011; aggregated_3: n=169, diff=0.981%, p=0.0277

### F5b_favorable_justices_over_unfavorable

Higher (hypothesis): `favorable_justices_to_lawyers`; lower: `unfavorable_justices_to_lawyers`. Entities: {'favorable_justices_to_lawyers': 1289, 'unfavorable_justices_to_lawyers': 1393}.

| Quantity | n (high, low) | mean high % | mean low % | diff % | Welch t | p | bootstrap 95% CI % | published |
|---|---|---:|---:|---:|---:|---:|---|---|
| aggregated_1 | [18, 53] | 2.374 | 2.625 | -0.251 | -0.27 | 0.7918 | [-2.076, 1.556] | * |
| aggregated_2 | [428, 706] | 1.905 | 2.258 | -0.354 | -1.84 | 0.0655 | [-0.726, 0.024] | ** |
| aggregated_3 | [428, 706] | 1.315 | 1.247 | 0.068 | 0.22 | 0.8265 | [-0.536, 0.663] |  |
| ppron | [299, 532] | 1.681 | 1.365 | 0.316 | 0.57 | 0.5674 | [-0.744, 1.377] |  |
| ipron | [335, 603] | 0.342 | 1.436 | -1.093 | -2.17 | 0.0304 | [-2.062, -0.112] |  |
| article | [293, 563] | 1.494 | 1.515 | -0.020 | -0.03 | 0.9738 | [-1.263, 1.175] |  |
| conj | [230, 480] | 3.152 | 1.947 | 1.205 | 1.68 | 0.0935 | [-0.213, 2.615] |  |
| prep | [309, 547] | 1.682 | 0.793 | 0.889 | 1.54 | 0.1237 | [-0.244, 2.027] |  |
| auxverb | [393, 665] | 1.065 | 0.758 | 0.308 | 0.79 | 0.4308 | [-0.463, 1.084] |  |
| adverb | [116, 254] | -0.276 | 3.343 | -3.619 | -2.88 | 0.0043 | [-6.084, -1.134] | ** |
| quant | [20, 60] | 6.094 | 6.911 | -0.816 | -0.19 | 0.8508 | [-8.360, 7.892] |  |

Paired sensitivity (entities present in both groups): aggregated_1: n=0, diff=n/a%, p=n/a; aggregated_2: n=237, diff=-0.463%, p=0.1127; aggregated_3: n=237, diff=-0.435%, p=0.3487

### F5a_pair_unit

Higher (hypothesis): `pair_lawyers_to_unfavorable`; lower: `pair_lawyers_to_favorable`. Entities: {'pair_lawyers_to_unfavorable': 1581, 'pair_lawyers_to_favorable': 1521}.

| Quantity | n (high, low) | mean high % | mean low % | diff % | Welch t | p | bootstrap 95% CI % | published |
|---|---|---:|---:|---:|---:|---:|---|---|
| aggregated_1 | [40, 9] | 3.758 | 2.032 | 1.726 | 1.06 | 0.3118 | [-1.352, 4.614] |  |
| aggregated_2 | [711, 393] | 3.466 | 2.505 | 0.961 | 5.09 | 0.0000 | [0.593, 1.336] |  |
| aggregated_3 | [711, 393] | 2.400 | 1.596 | 0.804 | 2.43 | 0.0154 | [0.153, 1.451] |  |
| ppron | [519, 253] | 3.487 | 2.265 | 1.221 | 2.12 | 0.0347 | [0.074, 2.351] |  |
| ipron | [583, 301] | 2.920 | 2.091 | 0.829 | 1.63 | 0.1041 | [-0.171, 1.812] |  |
| article | [534, 244] | 2.593 | 0.880 | 1.713 | 2.59 | 0.0099 | [0.425, 2.975] |  |
| conj | [442, 196] | 2.249 | 1.599 | 0.649 | 0.85 | 0.3942 | [-0.885, 2.138] |  |
| prep | [531, 269] | 2.424 | 1.225 | 1.199 | 2.00 | 0.0460 | [0.036, 2.365] |  |
| auxverb | [654, 359] | 1.842 | 1.342 | 0.500 | 1.25 | 0.2122 | [-0.292, 1.269] |  |
| adverb | [221, 82] | 3.203 | 3.776 | -0.573 | -0.36 | 0.7227 | [-3.724, 2.570] |  |
| quant | [49, 9] | 9.010 | 6.865 | 2.145 | 0.71 | 0.4902 | [-3.362, 7.948] |  |

### F5b_pair_unit

Higher (hypothesis): `pair_favorable_justices_to_lawyers`; lower: `pair_unfavorable_justices_to_lawyers`. Entities: {'pair_favorable_justices_to_lawyers': 1520, 'pair_unfavorable_justices_to_lawyers': 1578}.

| Quantity | n (high, low) | mean high % | mean low % | diff % | Welch t | p | bootstrap 95% CI % | published |
|---|---|---:|---:|---:|---:|---:|---|---|
| aggregated_1 | [10, 42] | 2.738 | 2.795 | -0.057 | -0.04 | 0.9687 | [-2.646, 2.687] |  |
| aggregated_2 | [379, 695] | 2.050 | 2.235 | -0.185 | -0.92 | 0.3572 | [-0.582, 0.216] |  |
| aggregated_3 | [379, 695] | 1.170 | 1.115 | 0.055 | 0.17 | 0.8679 | [-0.580, 0.718] |  |
| ppron | [245, 499] | 1.657 | 1.259 | 0.398 | 0.67 | 0.5002 | [-0.777, 1.542] |  |
| ipron | [290, 572] | -0.093 | 1.312 | -1.406 | -2.61 | 0.0092 | [-2.462, -0.379] |  |
| article | [250, 530] | 1.666 | 1.400 | 0.266 | 0.39 | 0.6936 | [-1.080, 1.589] |  |
| conj | [201, 439] | 3.167 | 1.962 | 1.205 | 1.49 | 0.1359 | [-0.363, 2.784] |  |
| prep | [266, 510] | 1.277 | 0.537 | 0.740 | 1.15 | 0.2496 | [-0.518, 1.989] |  |
| auxverb | [344, 654] | 1.117 | 0.677 | 0.440 | 1.07 | 0.2853 | [-0.365, 1.241] |  |
| adverb | [87, 222] | -0.613 | 3.215 | -3.828 | -2.81 | 0.0056 | [-6.448, -1.176] |  |
| quant | [12, 49] | 8.224 | 7.521 | 0.702 | 0.11 | 0.9169 | [-10.623, 13.667] |  |

### Sensitivities (aggregates only)

| Analysis | Claim | Agg 1 diff % (p) | Agg 2 diff % (p) | Agg 3 diff % (p) |
|---|---|---|---|---|
| notebook_supreme_3_3 | F4_lawyers_coordinate_more_than_justices | 0.727 (0.0009) | 0.833 (0.0000) | 0.996 (0.0000) |
| notebook_supreme_3_3 | fine_status_chief_over_associates | 0.751 (0.2467) | 0.336 (0.3423) | -0.521 (0.2386) |
| notebook_supreme_3_3 | F5a_lawyers_to_unfavorable_over_favorable | 1.000 (0.0025) | 0.934 (0.0020) | 0.943 (0.0037) |
| notebook_supreme_3_3 | F5b_favorable_justices_over_unfavorable | 0.399 (0.3641) | 0.038 (0.8576) | -0.185 (0.4963) |
| notebook_supreme_3_3 | F5a_pair_unit | 1.372 (0.0024) | 1.166 (0.0000) | 0.952 (0.0004) |
| notebook_supreme_3_3 | F5b_pair_unit | 0.233 (0.6198) | 0.035 (0.8610) | -0.082 (0.7584) |
| eq1_only_min_7_exchanges | F4_lawyers_coordinate_more_than_justices | 0.738 (0.0007) | 0.716 (0.0010) | 0.715 (0.0010) |
| eq1_only_min_7_exchanges | fine_status_chief_over_associates | 1.170 (0.0632) | 1.474 (0.0212) | 1.471 (0.0238) |
| eq1_only_min_7_exchanges | F5a_lawyers_to_unfavorable_over_favorable | 1.049 (0.0017) | 1.040 (0.0018) | 1.037 (0.0019) |
| eq1_only_min_7_exchanges | F5b_favorable_justices_over_unfavorable | 0.407 (0.2873) | 0.362 (0.3365) | 0.349 (0.3566) |
| eq1_only_min_7_exchanges | F5a_pair_unit | 1.142 (0.0038) | 1.065 (0.0059) | 1.066 (0.0066) |
| eq1_only_min_7_exchanges | F5b_pair_unit | 0.375 (0.3438) | 0.359 (0.3573) | 0.353 (0.3685) |
| exact_notebook_case_ids | F4_lawyers_coordinate_more_than_justices | 0.628 (0.0300) | 0.761 (0.0000) | 1.180 (0.0000) |
| exact_notebook_case_ids | fine_status_chief_over_associates | 0.994 (0.3815) | 0.208 (0.5405) | -0.376 (0.4717) |
| exact_notebook_case_ids | F5a_lawyers_to_unfavorable_over_favorable | 0.776 (0.0287) | 0.959 (0.0003) | 0.968 (0.0024) |
| exact_notebook_case_ids | F5b_favorable_justices_over_unfavorable | -0.404 (0.6693) | -0.383 (0.0468) | 0.067 (0.8292) |
| exact_notebook_case_ids | F5a_pair_unit | 1.774 (0.3002) | 0.920 (0.0000) | 0.795 (0.0171) |
| exact_notebook_case_ids | F5b_pair_unit | -0.255 (0.8603) | -0.223 (0.2688) | 0.057 (0.8652) |

## What reproduced and what did not

**F4, lawyers coordinate more toward Justices than Justices toward lawyers: reproduced.**
All three aggregates are positive and significant (Aggregated 1 +0.673 points, p=.020;
Aggregated 2 +0.765, p<.0001; Aggregated 3 +1.176, p<.0001), with bootstrap intervals
excluding zero. Six of eight markers point the published way and four are significant
(personal pronouns, impersonal pronouns, auxiliary verbs, adverbs). Articles (paper *) and
conjunctions point the other way, non-significantly; prepositions (paper ***) and
quantifiers (paper **) point the published way but are not significant here. The result
holds under all three sensitivities (p<=.03 for every aggregate).

**Fine status, lawyers coordinate more toward the Chief Justice than toward Associate
Justices (paper p<.01): not reproduced under the frozen protocol.** Aggregated 1 and 2 are
positive but far from significant (+1.013, p=.37, on 12 versus 371 entities; +0.225, p=.51),
Aggregated 3 is negative (-0.351, p=.50), and the paired sensitivity on the 145 lawyer
entities with both values is negative (Aggregated 3 -0.808, p=.10). Only auxiliary verbs
approach significance in the published direction (+1.038, p=.059). The published direction
appears only under the loosest sensitivity, plain Eq. (1) with at least seven exchanges
(Aggregated 2 +1.474, p=.021; Aggregated 3 +1.471, p=.024), and not under the 3/3 or literal
case-set variants. One of three sensitivities agreeing after the primary fails is not a
reproduction; the honest statement is that this contrast is threshold-dependent in these
data with this lexicon.

**F5a, lawyers coordinate more toward unfavorable than toward favorable Justices:
reproduced, and the most robust result here.** Aggregated 1 +0.799 (p=.024), Aggregated 2
+0.981 (p=.0002), Aggregated 3 +0.984 (p=.0019); all eight markers point the published way,
articles (+1.601, p=.0016) and quantifiers (+2.415, p=.041) significantly, prepositions and
conjunctions at p=.056 and .082. The paired comparison on 169 lawyer entities agrees
(Aggregated 2 +1.104, p=.0011), as do the pair-level unit (Aggregated 2 +0.961, p<.0001) and
all three sensitivities (every aggregate p<=.03). The paper's individual stars for personal
pronouns, impersonal pronouns and adverbs are not recovered.

**F5b, favorable Justices coordinate more toward lawyers than unfavorable Justices do:
not reproduced.** Aggregated 1 and 2 point the opposite way (-0.251, p=.79; -0.354, p=.066)
and Aggregated 3 is essentially zero (+0.068, p=.83). Two markers are significant in the
opposite direction: adverbs (-3.619, p=.0043; the paper marks adverbs ** in Figure 5(b), whose bars are not
printed in the text) and impersonal pronouns (-1.093, p=.030). The paired comparison on 237 Justice
entities is negative and non-significant, the pair-level unit repeats the two opposite-sign
markers, the 3/3 and Eq. (1) sensitivities give small positive non-significant aggregates,
and the literal case set gives a significant opposite-sign Aggregated 2 (-0.383, p=.047).
Nothing supports the published direction at the aggregate level.

In short: the two asymmetries that the paper presents as its headline Supreme Court results
(status, Figure 4; lawyers' dependence, Figure 5a) reproduce in direction and significance
with an unrelated function-word lexicon on public transcripts; the finer Chief Justice
contrast and the Justices' own dependence effect (Figure 5b) do not. Because the lexicon
differs, the failures could be lexicon-, threshold- or data-processing-dependent rather than
evidence against the original findings; this run cannot separate those explanations.

## Notes on `regesh.bonding.synchrony`

No bug was found. `CoordinationCounts` reproduces Eq. (1) exactly on constructed exchanges
with known conditional probabilities (test
`test_synthetic_conditional_probabilities_give_known_coordination`), and
`group_coordination` matches ConvoKit's Aggregated 1-3 definitions on the per-entity values.
One semantic divergence worth recording: when some marker has no group mean at all,
ConvoKit disables Aggregated 2 for the whole group, whereas `group_coordination` keeps the
entities that do not need that marker smoothed. Every marker has a group mean in all groups
here, so the two definitions coincide for this run. The measure has no thresholds of its
own; the ConvoKit-style minimum-count rule is applied by the reproduction runner on the
stored counts, without changing the library.

## Relevance to regesh

The synchrony module is meant as a measured accommodation channel. This reproduction shows
that the implemented measure, with its own lexicon, recovers the two large published
asymmetries in a real adversarial status hierarchy and fails to recover the two smaller
ones. It says nothing about whether coordination by an agent is desirable, about
subjective affect, or about the platform's controllers, whose saved weights are unchanged.
