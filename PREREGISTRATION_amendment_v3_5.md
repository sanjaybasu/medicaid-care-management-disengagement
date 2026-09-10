# Pre-registration amendment v3.5: bounded experiments on outreach effort, timing, channel, and who benefits

Dated 2026-09-10, written after the attempt log (encounter notes marked as not having occurred, with time of day) was pulled and before any of the results below were examined. Rules are fixed here; results are written only by the pipeline to `results/canonical.json`. Each experiment has a pass gate; an experiment that fails its gate is reported as a null.

## K1. Does the program's own outreach effort predict disengagement? (predictive)

Add to M2 four attempt features measured strictly before the decision point: logged attempts in the prior 30 and 90 days, attempts since the last completed contact, and days since the last attempt. Refit on the training set and evaluate on the test set with the same split and learner as the main analysis. Gate: AUROC gain over M2 with a 95% patient-bootstrap interval excluding zero. Also report the gain of the full model (M5) plus attempts over M5.

## K2. Does continued attempting after a completed contact bring the patient back? (within-patient dose response)

Unit: eligible decision point t with no completed contact in days 1 to 7. Exposure: logged attempts in days 1 to 7 (0, 1, 2 or more). Outcome: a completed contact in days 8 to 30. Estimator: linear probability model with patient fixed effects (patients contribute several decision points at different attempt intensities), market by quarter fixed effects, and time-varying covariates (days since activation, contacts in the prior 90 days, attempts in the prior 30 days), standard errors clustered by patient; marginal structural co-estimator on the same sample. Gate 1 (pre-trend): within patients, the exposure must not predict the number of completed contacts in the 30 days before t (standardized difference at most 0.10 per exposure level). Gate 2: within-patient and marginal estimates agree in sign with overlapping intervals. Pass: both gates and an interval excluding zero.

## K3. When do attempts succeed? (timing, within patient)

Unit: logged attempt (call, text, or visit that did not reach the patient). Outcome: a completed contact within 48 hours after the attempt. Exposures: local time block of the attempt (8 am to noon; noon to 5 pm; 5 pm to 9 pm; other), weekday versus weekend, channel (call, text, visit, other), and role of the staff member. Estimator: linear probability model with patient fixed effects and staff-role, market, and quarter fixed effects, standard errors clustered by patient, so that the same patient's attempts at different times and by different channels are compared. Gate: pre-trend placebo, in which time block must not predict the patient's completed contacts in the prior 30 days within patient (standardized difference at most 0.10). Heterogeneity: by age band (under 45; 45 and over) and by prior response pattern (whether the patient had ever completed an evening or weekend contact). Pass: a time block, channel, or day effect with an interval excluding zero that survives the placebo.

## K4. Who benefits from an early second contact? (heterogeneous effect of the v3.1 lever L5)

Sample: the equipoise sample of the L5 lever (second completed contact within 7 days of the first). Estimator: doubly robust learner of the conditional effect on disengagement in days 31 to 120 with cross-fitting, using baseline covariates and first-contact features. Validation: split-sample rank test. The conditional effect is estimated on a random half, patients in the other half are ranked by the predicted effect, and the doubly robust average effect in the top half of the ranking is compared with the bottom half; the gate is a difference with a 95% bootstrap interval excluding zero (a targeting rule that beats uniform treatment). Report the characteristics of the top-ranked half.

## K5. Does effort bundled with contact predict better than contact alone? (descriptive)

Report the disengagement rate by joint level of completed contacts and logged attempts in the first 30 days, to show whether attempts carry information beyond completed contacts.

## Reporting

One supplementary table per experiment; a main-text paragraph only for experiments that pass their gate; failures reported as nulls in the Supplement.
