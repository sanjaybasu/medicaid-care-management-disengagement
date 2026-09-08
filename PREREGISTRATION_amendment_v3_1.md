# Pre-registration amendment v3.1: pseudo-prospective validation, physician review, simple score, additional levers, capacity, and fairness

Dated 2026-09-08, before any of the analyses below were run. Amends `PREREGISTRATION_v3_disengagement.md`. Rules are fixed here; results are written only by the pipeline to `results/canonical.json`.

## A. Pseudo-prospective (frozen-model, rolling-landmark) validation

Purpose. Emulate a shadow deployment retrospectively: freeze the model using only information available at a landmark date L, then score every contact after L as a deployed system would have and report realized disengagement.

Design. Landmarks are month-ends from 2025-01-31 to 2025-12-31. At each L, the training set is every eligible decision point with contact date on or before L minus 90 days (so that its 90-day outcome was observable at L), from all members regardless of era. The frozen model is the full model without embeddings (structured, contact history, lexicon, tags, and the TF-IDF note stack; M3b), chosen because it can be refit at every landmark within compute limits and differs from M5 by 0.014 in AUPRC in the main analysis. The scoring set is every eligible decision point with contact date after L (through 2026-06-07, the last date with complete 90-day follow-up). The flagging threshold is fixed at L as the 80th percentile of predicted probability in the training set, and decile cutpoints for calibration are fixed from the training set.

Reported at each L: training decision points and events; scored decision points and events; AUROC, AUPRC, calibration slope and intercept; realized disengagement by training-defined decile; share of scored contacts flagged and disengagements per 100 flags at the fixed threshold; the same metrics by calendar month since L (months 1 to 6 and beyond) and separately for members who had contributed training contacts and members who had not.

Choice of the headline landmark (rule fixed before results). Among landmarks that leave at least six full calendar months of scored contacts with complete follow-up (L on or before 2025-11-30) and whose training set contains at least 2,000 disengagement events, we select the latest. The full rolling-landmark curve is reported regardless, as a figure, so that the choice does not hide decay.

What this does and does not show. It shows whether a model fixed at L discriminates and stays calibrated on contacts scored afterwards, and how fast it decays without refitting. It does not show the effect of acting on a flag, because no one acted on it; that requires the randomized step.

## B. Blinded physician review

Purpose. Establish what disengagement is clinically (needs met, dissatisfaction, loss of contact, life event, administrative) and whether it carries clinical stakes; measure physician prognosis against the model on the same cases; check the face validity of the narrative patterns.

Sample. 200 test-era decision points scored by the full model, stratified by predicted-risk stratum (top decile; deciles 4 to 7; bottom three deciles) and by realized outcome (disengaged, retained), about 33 per cell, with random selection within cells (seed 20260908). Each case packet contains the index note, the notes of the prior 90 days, and a structured summary (age band, gender, state, discipline and encounter type of the index contact, days since activation, contacts in the prior 90 days, condition flags). Packets contain nothing after the index contact, no model score, and no outcome. Three physicians review the cases in a balanced incomplete design: each case is reviewed independently by two of the three physicians (pairs assigned at random with equal counts), and a core set of 30 cases, drawn across all six cells, is reviewed by all three; agreement is reported pairwise on all cases and three-way on the core set.

Phase 1 (blinded) items per case: (1) engagement status at the index contact (engaged, ambivalent, declining, unreachable, administrative only); (2) physician prognosis of a completed contact within 90 days (five-point scale); (3) open clinical need at the index contact (medication gap, pending referral or appointment, uncontrolled condition, active social crisis, behavioral health need, none), multiple allowed; (4) harm if lost to the program (low, moderate, high); (5) explicit statement of intent in the notes (wants to continue, wants fewer contacts, wants to stop, none).

Phase 2 (unblinded, disengaged cases only, about 100, one physician per case with a 20% double-coded subset): with subsequent notes and status history shown, adjudicate the reason (needs met or graduated in substance; dissatisfied or declined; lost contact without stated reason; moved, incarcerated, hospitalized, or deceased; administrative or coverage; cannot determine).

Phase 3 (pattern review): the 20 narrative patterns with the largest univariate association with disengagement (lexicon indicators and top TF-IDF terms), each with three de-identified example snippets; rate clinical meaningfulness (1 to 5) and note what the pattern indicates.

Analysis. Inter-rater agreement by Fleiss kappa for categorical items and intraclass correlation for scales; physician prognosis AUROC against realized disengagement compared with the full model's AUROC on the same 200 cases with a paired bootstrap; distribution of adjudicated reasons; prevalence of open clinical need and high harm among disengaged versus retained cases; agreement between narrative-pattern ratings and univariate rates. A quantitative companion on all test-era members compares 90-day acute care events (emergency department visits and inpatient admissions from admission-discharge-transfer feeds) between disengaged and retained decision points, adjusted for the baseline risk percentile and days since activation.

## C. Simple points score

A logistic regression on at most eight pre-chosen predictors (contacts in the prior 90 days, days since the last contact, first 30 days after activation, encounter type recorded as patient outreach, contact by a care coordinator or pharmacy technician, the declined lexicon pattern in the current note, the positive-engagement pattern in the current note, open goals), fit on the training era, with coefficients rounded to integer points (largest coefficient scaled to 10 points; log-transformed counts enter as their log value times the points). Reported: AUROC, AUPRC, calibration, net benefit at 20% flagged on the test era, and the share of the full model's gain over M0 that the points score captures.

## D. Additional levers

L5: time from the first outreach-phase contact to the next completed contact (at most 7 days versus longer), exposure defined at the first contact; outcome disengagement in days 31 to 120 after the second contact. L6: same-person continuity, defined as the next contact made by the same care team member as the index contact versus a different one; outcome disengagement in days 31 to 120 after the next contact. Same estimators, gates, equipoise rule, and sensitivity bounds as Section 7 of v3. Lever eligibility now also requires the decision point to be on or before 2026-05-09 so the 120-day window is complete.

## E. Capacity arithmetic

At weekly re-engagement capacities of 5, 10, 20, and 40 attempts per care team (teams defined by market), the number of true disengagements reached per week under four allocation rules: the full model's flags, the deployed risk score, all contacts in the first 30 days after activation, and random selection; computed on the test era by week and averaged.

## F. Fairness

By sex, race and ethnicity, state, age band (under 18, 18 to 44, 45 to 64, 65 and over), behavioral health diagnosis, and preferred language where populated: event rate, AUROC, AUPRC, calibration slope and intercept, and, at the single overall 20% threshold, sensitivity, positive predictive value, and flag rate, for the full model and for M2. Differences from the reference group with member-bootstrap intervals.

## G. Additions of 2026-09-08 (afternoon), specified before their results were examined

Metric set and intervals. For every model on the test era: AUROC, AUPRC, Brier score, calibration slope and intercept, and at the model's own 80th-percentile threshold sensitivity, specificity, positive and negative predictive value, F1, and net benefit, each with a 95% percentile interval from 1,000 member-bootstrap resamples; a decision-point bootstrap for the full model to compare interval width; subgroup AUROC and AUPRC intervals by state and by race and ethnicity (500 resamples within subgroup).

Ablation. Leave-one-group-out and single-group fits of the full model over five predictor blocks (structured data; contact history; lexicon and tags; note embeddings; TF-IDF note stack); an L2-penalized logistic regression (C = 0.1, standardized features) on the same features; gradient boosting with 7 and 31 maximum leaves and with learning rate 0.1.

Quarterly refitting. The full model without embeddings retrained at each quarter-end from 2025-03-31 to 2026-03-31 on decision points dated at least 90 days earlier and applied to the following quarter; the concatenated series compared with a single model frozen at 2025-03-31 on the same pooled decision points (AUROC, AUPRC, calibration slope and intercept, flag rate, disengagements per 100 flags, sensitivity; 500 member-bootstrap resamples); member-bootstrap intervals for the headline frozen landmark.
