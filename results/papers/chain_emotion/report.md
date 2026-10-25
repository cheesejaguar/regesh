# Chain-of-Emotion: statistical reproduction

Source: Croissant et al. (2024), [PLOS ONE](https://doi.org/10.1371/journal.pone.0301033); [author data](https://doi.org/10.17605/OSF.IO/QPT6Z).

This re-executes analyses on the authors' original outputs and participant ratings. It does not rerun GPT-3.5, recruit participants, or recompute proprietary LIWC dictionaries.

## Experiment 1: original 42-item STEU outcomes

| Condition | Published | Reproduced | Accuracy |
|---|---:|---:|---:|
| No Memory | 24/42 | 24/42 | 0.5714 |
| Memory | 31/42 | 31/42 | 0.7381 |
| Appraisal Prompts | 35/42 | 35/42 | 0.8333 |

Paired exact McNemar tests use the same items and Holm-correct three contrasts:

- Memory minus No Memory: +0.1667; paired bootstrap 95% CI [0.023809523809523808, 0.30952380952380953]; exact p=0.0654297, Holm p=0.130859.
- Appraisal Prompts minus No Memory: +0.2619; paired bootstrap 95% CI [0.11904761904761904, 0.40476190476190477]; exact p=0.00341797, Holm p=0.0102539.
- Appraisal Prompts minus Memory: +0.0952; paired bootstrap 95% CI [0.0, 0.21428571428571427]; exact p=0.21875, Holm p=0.21875.

## Experiment 2: released LIWC scores

| Metric | Published F (p) | Reproduced numeric 0/1/2 trend F (p) | Categorical F (p) |
|---|---:|---:|---:|
| WC | 0.20 (0.65) | 0.20296 (0.65372) | 1.08077 (0.34493) |
| Authentic | 5.10 (0.03) | 5.09332 (0.02710) | 3.30174 (0.04263) |
| Tone | 2.76 (0.10) | 2.75752 (0.10121) | 1.37963 (0.25843) |
| Affect | 0.08 (0.78) | 0.08070 (0.77718) | 0.60697 (0.54784) |
| emo_pos | 1.76 (0.19) | 1.76450 (0.18832) | 0.87816 (0.42007) |
| emo_neg | 0.37 (0.55) | 0.36789 (0.54609) | 1.61556 (0.20610) |

Complete-metric filtering retains 73/75 sentence rows (22/24/27). The two excluded Chain-of-Emotion rows contain only punctuation. Numeric trend coding reproduces the reported F pattern, whereas a categorical condition test gives different statistics. All six trend tests fail Holm correction at .05. Sentences from only three generated conversations do not support an independent population-level replication.

## Experiment 3: all released participant ratings

The published description calls this a within-subject design; this analysis uses the 30 participant IDs and all 90 rows. Published Table 3 is not reproduced exactly. Raw ratings span 1–7, while the table caption says 0–6. Published residual df=84 differs from full-data independent df=87 and repeated-measures df=58; the source code/exclusion rule is unavailable.

| Item | Published F (p) | Full-data unpaired diagnostic F (p) | Paired RM F (p; Holm p) |
|---|---:|---:|---:|
| The agent's behavior was human-like. | 1.95 (0.15) | 1.7097 (0.1869) | 4.7500 (0.0123; 0.1106) |
| The agent's reactions were natural. | 3.65 (0.03) | 4.1039 (0.0198) | 7.7042 (0.0011; 0.0129) |
| The agent reacted to my input. | 3.62 (0.04) | 4.3246 (0.0162) | 6.0825 (0.0040; 0.0440) |
| The agent did not care about the scenario. | 0.26 (0.78) | 0.1169 (0.8898) | 0.3584 (0.7004; 1.0000) |
| The agent always knows their friends' emotions from their behavior. | 1.32 (0.27) | 2.2250 (0.1142) | 4.1149 (0.0213; 0.1706) |
| The agent is a good observer of others' emotions. | 2.25 (0.11) | 2.2931 (0.1070) | 2.9753 (0.0589; 0.4121) |
| The agent is sensitive to the feelings and emotions of others. | 3.31 (0.04) | 3.7871 (0.0265) | 5.5187 (0.0064; 0.0640) |
| The agent has a good understanding of the emotions of people around them. | 0.86 (0.43) | 1.2401 (0.2944) | 1.5310 (0.2249; 1.0000) |
| How capable was the agent? | 1.68 (0.19) | 0.9939 (0.3743) | 0.9926 (0.3768; 1.0000) |
| How competent was the agent? | 1.68 (0.19) | 0.6046 (0.5486) | 0.6956 (0.5029; 1.0000) |
| How warm was the agent? | 2.48 (0.09) | 1.1724 (0.3145) | 2.2855 (0.1108; 0.6649) |
| How friendly was the agent? | 0.41 (0.66) | 0.6955 (0.5016) | 1.3724 (0.2616; 1.0000) |

Paired contrasts, participant bootstrap intervals, standardized differences, Greenhouse–Geisser sensitivity and Holm corrections are in summary.json. These are exploratory reanalyses of an existing small study, not preregistered confirmations or evidence of subjective experience. The strongest appraisal-versus-no-memory STEU result reproduces; subjective-rating claims require caution about analysis choices and multiple outcomes.
