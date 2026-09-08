# Pre-registration v3: Predicting and preventing disengagement from Medicaid care management using the multidisciplinary care record

Version 3.0, 2026-09-07. New project (`packaging/care-management-disengagement/`, `notebooks/care-management-disengagement/`). Data caches inherited from the sequence-NBA project (symlinked `data_cache/`, pulled 2026-09-06, WA/VA/OH, activations 2023-05-23 to 2026-06-05, contacts through 2026-06-07). Owner: Sanjay Basu. Reviewers before OSF deposit: Parth Sheth, Benjamin Huynh, Sara Greenbaum (operations), Aaron Baum.

Rule: nothing below is a result. Numbers describe the data before analysis. Results are written only by the pipeline to `results/canonical.json`; every table and figure renders from that file; `audit_consistency.py` must report zero discrepancies before any document leaves the repository. Deviations are logged in Section 10 with dates.

---

## 1. Why this study

The program's published effects are conditional on members staying engaged: activation reduces acute care (target-trial, JAMA Network Open), effect-based targeting prevents more events per unit of capacity (Population Health Management 2026; Health Services Research 2026), and next-step recommendations can be learned from tabular states (JMIR AI 2025). None of those studies asked who stops engaging, when, or whether the care record can see it coming. Operationally, disengagement is the binding constraint: in Virginia, 59% of activated members had one or fewer contact-days in their first 90 days, 77% of activation spans ended by administrative reset rather than graduation, and incentive changes aimed at deepening engagement since Q3 2025 have not moved these figures. Nobody has tested whether what care teams write predicts who will disappear.

Pre-analysis evidence that motivated this study (temporal test era, 18,187 decision points): a structured-data model predicts 90-day loss of contact with AUROC 0.605; adding contact-sequence features and language-model tags gives 0.633; adding note text gives 0.670, with text alone at 0.650. For 90-day acute events the same additions gave nothing (0.737 with or without notes). The record's unique information is about engagement, not deterioration.

## 2. Questions and hypotheses

Primary question. At each completed care-team contact, can the member's care record, including note text, predict whether the member will disengage within 90 days, and how much does the narrative record add to structured data?

H1. The model with note text has higher discrimination (AUPRC, AUROC) and net benefit for 90-day disengagement than the structured-data model, on the temporally forward, patient-disjoint test era.
H2. The narrative contribution is present in every state and every racial and ethnic group (subgroup AUROC gains all above zero).
H3 (identified lever). Within patients, an in-person contact at t is followed by a lower probability of disengagement in days 31 to 120 than a remote contact at t, after the pre-trend gate; the same is tested for a completed pharmacist review and a completed therapy session.
H0 statements reported if observed: if H1 fails, the paper reports the structured model and the null value of notes; if H3 fails its gates, the lever section reports bounds and the paper stands on prediction.

## 3. Design summary

Retrospective prognostic study at the decision-point level with temporally forward validation and TRIPOD+AI reporting; an identified within-patient analysis of contact modality and discipline as levers; a registered prospective step.

## 4. Population, decision points, and outcomes

Population. Activated members in Washington, Virginia, and Ohio as in the inherited decision-point file: 52,961 completed contacts within 365 days of activation, 8,853 patients; training era activations through 2025-06-30 (34,774 contacts), test era after (18,187 contacts, 3,601 patients).

Analytic eligibility for outcomes. At least 60 enrolled days in the 90 days after the contact (49,834 of 52,961 decision points, 94.1%), so that absence of contact cannot be explained by loss of coverage. Sensitivity: all decision points with enrollment as a covariate.

Primary outcome, disengagement within 90 days: no completed care-team contact in days 1 to 90 after t and no graduation status in that window. Pre-analysis rate 18.5% (test era 18.8%); by state Ohio 13.2%, Virginia 22.6%, Washington 13.3%; by days since activation 25.4% (0 to 30), 8.1% (31 to 90), 10.0% (91 to 180), 16.4% (181 to 365).

Secondary outcomes. (a) Explicit patient-driven exit within 90 days: a status of dropped out of contact, withdrawn by patient, or refused (rate 9.5%, 4,753 events). (b) Silent loss: the primary event with no administrative status change in the window (46.2% of primary events), which isolates disengagement not explained by roster resets. (c) No contact within 30 days (rate 28.8%).

Graduation within 90 days (32.6% of decision points) is a planned exit and is never counted as disengagement. Withdrawal by Waymark (1.3%) is excluded from the primary event and reported.

## 5. Predictors (all measured strictly before or at t)

Structured: age, gender, race and ethnicity, state and market, Signal risk percentile and tier at activation, condition flags, days since activation, prior acute events (30, 90, 180, 365 days), days since last acute event, enrollment history.
Contact history: counts by discipline and modality in the prior 30 and 90 days, cumulative contacts, days since last contact, mean and variability of inter-contact gaps, number of distinct disciplines, open and completed goals by category.
Current contact: discipline of author, modality, encounter type.
Narrative: (a) TF-IDF unigrams and bigrams over the current note and over prior 90-day notes (baseline, fully interpretable); (b) dense embeddings of the current note and mean-pooled prior 90-day notes from a general-domain sentence-embedding model run locally (BAAI/bge-base-en-v1.5; no note leaves the machine), reduced to 64 dimensions by PCA fit on the training era; (c) a pre-specified engagement lexicon (unable to reach, no answer, voicemail, wrong or disconnected number, declined, not interested, requested fewer or no calls, moved, incarcerated, hospitalized, busy, will call back) as binary indicators, for interpretability; (d) the five existing language-model action tags. Any new language-model tagging is documented in Methods with model, prompt version, and a 200-note clinician validation sample.

## 6. Models and evaluation

Models (all trained on the training era, hyperparameters frozen from a 20% pilot draw with seed 20260907 before test-era evaluation): M0 deployed Signal risk percentile alone (status quo reference); M1 structured; M2 structured plus contact history; M3 M2 plus narrative TF-IDF stacked logit; M4 M2 plus dense embeddings; M5 M4 plus lexicon and tags (full model); M6 sequence models (GRU, Transformer, state-space) over the last ten contacts' tokens at equal parameter budgets, three seeds, early stopping, reported as the architecture comparison. Gradient-boosted trees are the base learner for M1 to M5.

Metrics on the test era: AUROC, AUPRC, calibration slope and intercept and calibration plot, Brier score, net benefit at flagging 10%, 20%, and 30% of contacts (decision curve), sensitivity and positive predictive value at those operating points, and the number of disengagements per 100 flags. Differences between models with patient-cluster bootstrap intervals (1,000 resamples). Primary contrast: M5 versus M2 on AUPRC and net benefit at 20% flagged. Secondary: M2 versus M0.

Validation beyond the temporal split: leave-one-state-out (train two states, test the third); by race and ethnicity, language, state, days-since-activation stratum, and activating discipline. Interpretability: SHAP for M5; top TF-IDF terms and lexicon indicators; a table of the 20 highest-risk narrative patterns reviewed by two care-team leads.

## 7. Identified lever: what a team does at t and disengagement afterwards

Exposures at t (each binary, analysed separately): L1 in-person contact versus remote; L2 pharmacist review versus not; L3 therapy session versus not; L4 contact by a discipline different from the previous contact versus same discipline.
Outcome: no completed contact in days 31 to 120 after t (excluding days 1 to 30 so the exposure is not part of its own outcome), not graduated, at least 60 enrolled days.
Identification: within-patient conditional logistic regression on switchers, calendar-month and days-since-activation adjustment, time-varying covariates measured before t; history-adjusted marginal structural weights (unclipped propensities, weights truncated at the 1st and 99th percentiles) as co-estimator; equipoise trimming to propensity in [0.10, 0.90]. Gates as in the sequence-NBA v2 specification: G1 flat pre-period engagement bins (three 30-day bins before t) with the equivalence margin of 10% of the base rate; G2 estimator agreement; failure is reported with Kallus–Zhou bounds. Timing stratification by days since activation. Minimum detectable effects computed on clustered variance before estimation.

## 8. Prospective step (registered)

Shadow deployment for 90 days: the M5 flag is computed at each contact in Virginia and Washington and stored, not shown; realized disengagement by flag decile is the prospective calibration. Then a cluster-randomized trial by care team of a re-engagement protocol for flagged members (second attempt within 7 days by a different modality, then in-person attempt), outcome 90-day disengagement; power from the observed 18.5% base rate, the test-era intra-team correlation, and the number of teams.

## 9. Reporting

npj Digital Medicine Article: abstract up to 150 words unstructured; Introduction and Discussion without subheadings; Results and Methods with matching subheadings in this order: Cohort and disengagement; Prediction; Value of the narrative record; Subgroups and states; Levers; supplementary architecture comparison. Every estimate with interval and denominator; "disengagement" used once defined and never varied; no evaluative adjectives. TRIPOD+AI checklist as Supplement S1; pre-registration as S2 with deviation log; code release with DOI and synthetic example data; language-model use documented in Methods; WCG IRB 20253751 with 45 CFR 46.116(f) and 164.512(i); competing interests in the standard Waymark sentence; funding none. Manuscript and figures in `notebooks/care-management-disengagement/`; code and results in `packaging/care-management-disengagement/`.

Display items: Figure 1 flow and decision-point schematic; Figure 2 discrimination and calibration by model; Figure 3 decision curves and disengagements per 100 flags; Figure 4 subgroup and state performance; Figure 5 narrative patterns and SHAP; Figure 6 lever event studies. Tables: 1 cohort; 2 model performance with intervals; 3 subgroup and leave-one-state-out; 4 lever effects with gates; 5 sensitivity grid (eligibility threshold 60 versus 90 days, outcome definitions a to c, embedding model, prior-note window 30 versus 90 days, era split, bootstrap level).

## 10. Deviation log

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-09-07 | 6 | M6 sequence models used 32 (not 64) embedding principal components per contact token and nine structured covariates; reported as the supplementary architecture comparison. | Token dimension kept small at equal parameter budgets. |
| 2026-09-07 | 6 | Interpretability used permutation importance on the full model without embeddings and univariate lexicon rates instead of SHAP; the 20-pattern clinician review table was not produced. | Permutation importance is model-agnostic and reproducible from the results file; clinician review deferred to the prospective step. |
| 2026-09-07 | 6 | Subgroup analysis by language was not performed. | Preferred language is not populated in the decision-point file. |
| 2026-09-07 | 7 | L2 (pharmacist review), L3 (therapy session), and L4 (discipline change) were not estimable: 0, 0, and 87 decision points in the equipoise region. | Pre-specified rule of at least 200 decision points and 50 exposed in the propensity range 0.10 to 0.90. |
| 2026-09-07 | 9 | The sensitivity grid (Table 5 in the plan) was reduced to: eligibility 60 versus 90 enrolled days, no eligibility restriction with enrolled days as a covariate, training contacts restricted to before 2025-07-01, and the three secondary outcomes, each for M2 and M3; the embedding-model, prior-note-window, and bootstrap-level variations were not run. Display-item numbering: Table 5 is the architecture comparison and the sensitivity grid is Supplementary Table S3. | Compute time; the retained variations address the eligibility and calendar-overlap concerns raised in review. |
| 2026-09-07 | 4 | The training/test split is by activation date (patient-disjoint, forward in activation); contacts of training-era members after 2025-07-01 remain in training. | As specified in Section 4; documented as a limitation and tested in the sensitivity grid. |

## 11. Engineering

`scripts/01_outcomes.py` (outcomes and eligibility from contacts, status history, enrollment spans), `02_features.py` (structured, history, lexicon, tags), `03_embed.py` (bge-base embeddings, PCA on training era), `04_models.py` (M0 to M6, pilot tuning, frozen hyperparameters, test-era evaluation, bootstrap), `05_subgroups.py`, `06_levers.py` (within-patient and MSM with gates), `07_report.py` (render from `results/canonical.json`), `audit_consistency.py`. No script writes a literal result.
