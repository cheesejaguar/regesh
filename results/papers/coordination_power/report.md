# Echoes of power, Supreme Court results: machine-written tables

Source: [https://arxiv.org/abs/1112.3670](https://arxiv.org/abs/1112.3670). Coordination values are percent (x100). The lexicon is an authored open list, not LIWC: compare directions and significance, not levels.

Selected sessions: 204 (201 case ids); exchanges kept: 49281; Justices: 11; lawyers: 312.

## F4_lawyers_coordinate_more_than_justices

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

## fine_status_chief_over_associates

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

## F5a_lawyers_to_unfavorable_over_favorable

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

## F5b_favorable_justices_over_unfavorable

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

## F5a_pair_unit

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

## F5b_pair_unit

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

## Sensitivities (aggregates only)

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
