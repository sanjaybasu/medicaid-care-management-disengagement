# Predicting disengagement from Medicaid care management from the multidisciplinary care record

Public repository: https://github.com/sanjaybasu/medicaid-care-management-disengagement

Analysis code for the manuscript of the same title (target: npj Digital Medicine) and the accompanying operations memo. Code only: no patient data, notes, embeddings, or predictions are stored in this repository. `data_cache/` is a local symlink to the protected data environment and is not versioned.

## Pipeline

| Step | Script | Writes |
|---|---|---|
| 1 | `scripts/01_outcomes.py` | `data_cache/dp_outcomes.parquet`, `results/flow.json` (eligibility, primary and secondary outcomes, lever outcome) |
| 2 | `scripts/02_features.py` | `data_cache/dp_features.parquet`, `results/features_meta.json` (structured, contact-history, lexicon, tag features) |
| 3 | `scripts/03_embed.py` | `data_cache/note_emb_bge.parquet` (bge-base-en-v1.5 note embeddings, computed locally) |
| 4 | `scripts/04_models.py --with-embeddings` | `results/models_primary.json`, `results/hyperparameters.json`, `data_cache/test_predictions.parquet` (M0 to M5, contrasts with member-bootstrap intervals, secondary outcomes) |
| 5 | `scripts/05_subgroups.py` | `results/subgroups.json` (subgroups, leave-one-state-out, permutation importance, lexicon rates) |
| 6 | `scripts/06_levers.py` | `results/levers.json` (within-member conditional logit, marginal structural model, gates, sensitivity bounds) |
| 7 | `scripts/07_sequence.py` | `results/sequence_models.json` (GRU, Transformer, Mamba-form state-space model) |
| 8 | `scripts/08_report.py` | `results/canonical.json`; figures and tables in `../../notebooks/care-management-disengagement/` |
| 9 | `scripts/09_sensitivity.py` | `results/sensitivity.json` (eligibility, calendar split, outcome definition) |
| 10 | `scripts/10_landmark.py` | `results/landmark.json` (frozen-model rolling-landmark validation) |
| 11 | `scripts/11_physician_sample.py` | blinded physician review packets and forms in `data_cache/physician_review/` (PHI, local only); `results/physician_sample_meta.json` (counts) |
| 12 | `scripts/12_physician_analysis.py` | `results/physician_review.json` (run after forms are returned) |
| 13 | `scripts/13_simple_score.py` | `results/simple_score.json` (eight-predictor points score) |
| 14 | `scripts/14_levers2.py` | `results/levers2.json` (second contact within 7 days; same-person continuity) |
| 15 | `scripts/15_capacity.py` | `results/capacity.json` (re-engagement capacity arithmetic) |
| 16 | `scripts/16_fairness.py` | `results/fairness.json` (subgroup fairness at a single threshold) |
| 17 | `scripts/17_clinical_stakes.py` | `results/clinical_stakes.json` (acute care after disengagement) |
| 18 | `scripts/18_metric_cis.py` | `results/metric_cis.json` (full metric set with member-bootstrap intervals; subgroup intervals) |
| 19 | `scripts/19_ablation.py` | `results/ablation.json` (leave-one-group-out, single-group, logistic regression, hyperparameter variants) |
| 20 | `scripts/20_rolling_refit.py` | `results/rolling_refit.json` (quarterly refit versus frozen model) |
| (dry run) | `scripts/adjudicate_reviewer_a.py`, `scripts/test_reviewer_a_adjudication.py` | keyword-rule procedure used to pre-fill one physician reviewer's forms before that reviewer's case-by-case review and approval (documented in Supplementary Note S7) |
| 21 | `scripts/21_program_scope.py` | `results/program_scope.json` (distinct assigned primary care clinicians, practice organizations, and partner entities for the eligible members; read-only query of the clinical data mart) |
| 22 | `scripts/22_table_cis.py` | `results/table_cis.json` (member-bootstrap intervals for every Table 3 subgroup and leave-one-state-out row, and for the secondary outcomes in Supplementary Table S18) |
| audit | `audit_consistency.py` | Discrepancy table comparing every number in the manuscript, memo, and tables to `results/canonical.json`; exit code 1 on any mismatch |

Run in order from this directory with the project Python environment. No script writes a literal result; every figure and table is rendered from `results/canonical.json`.

## Reproducibility

Every figure and table in the manuscript is rendered by `scripts/08_report.py` from `results/canonical.json`, which is written locally by the pipeline; no results, data, notes, or embeddings are stored in this repository. `audit_consistency.py` exits non-zero if any number in the manuscript, memo, or tables cannot be derived from that file.

Program-scope descriptors in the manuscript (contracted clinicians, practice organizations, and partner entities) come from the program's internal provider roster as of 20 January 2026, recorded in the local results file and not released.

## Pre-registration

`PREREGISTRATION_v3_disengagement.md` and `PREREGISTRATION_amendment_v3_1.md` (dated 2026-09-08) fixed the outcome, predictors, models, contrasts, validation design, lever gates, and reporting plan before test-era analysis. Section 10 records deviations with dates.

## Ethics

WCG Institutional Review Board protocol 20253751; waiver of informed consent under 45 CFR 46.116(f) and waiver of HIPAA authorization under 45 CFR 164.512(i).
