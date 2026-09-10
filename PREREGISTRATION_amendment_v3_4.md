# Pre-registration amendment v3.4: assigned-staff effects on disengagement

Dated 2026-09-10, written after the v3.3 instruments failed Gate 4 (each staff member's rate of one action tracked the rates of the others, so the exclusion restriction does not hold) and before any staff-effect result was examined. Rules are fixed here; results are written only by the pipeline to `results/canonical.json`.

## J1. Question

How much of a patient's risk of disengagement in days 31 to 120 after activation depends on which care team member made the first contact, holding market, calendar quarter, and baseline characteristics fixed, and does the staff member's effect differ by patient characteristics?

## J2. Data and sample

Same patient-level file as v3.3: one row per patient; assigned staff member = the staff member who made the first completed contact within 30 days of activation; outcome = no completed contact in days 31 to 120 and no program completion by day 120; patients enrolled for all of days 31 to 120; staff members with at least 20 assigned patients.

## J3. Estimation

Linear probability model of the outcome on market by activation-quarter fixed effects, baseline covariates, and staff fixed effects, with staff-clustered standard errors. Staff effects are centered within market and shrunk by empirical Bayes (random-intercept mixed model). Reported: the standard deviation of the shrunken staff effects on the probability scale; the intraclass correlation; the mean predicted disengagement for patients assigned to staff in the lowest and highest quartile of the effect distribution; the share of staff whose effect differs from the market mean with 95% confidence.

## J4. Validation, fixed before results

Split sample. Each staff member's patients are divided at random (seed 20260910) into halves A and B. Staff effects are estimated on A and used as a predictor of B outcomes in the same fixed-effects model with covariates. A coefficient near 1 (95% CI including 1 and excluding 0) indicates the effect forecasts out of sample and is not noise; a coefficient near 0 indicates noise. Balance: the A-estimated staff effect must not predict B patients' baseline covariates (standardized differences at most 0.10 per SD; joint p above 0.05), which would indicate that assignment tracks patient traits.

## J5. Heterogeneity and mechanism

Interaction of the A-estimated staff effect with B patients' age (under 45 versus 45 and over), behavioral health diagnosis, risk percentile above the median, prior acute care event, state, and race and ethnicity, to ask whether the same staff members help all patients or particular ones. Descriptively, the association of each staff member's effect with the staff member's action profile (mean contacts per patient in days 0 to 30; in-person, weekend, and 7-day follow-up rates; role), reported as correlations without a causal reading, because v3.3 showed these actions travel together.

## J6. Reporting

One supplementary table for the v3.3 instruments with gate results (reported as failing Gate 4), one for the staff effects with validation and heterogeneity, a main-text paragraph, and a Discussion statement that the actions form a bundle that observational data cannot separate and that a micro-randomized trial of the components is the appropriate next design.

## J7. Role confounding (added 2026-09-10 after the all-staff model, before the within-role results were examined)

The all-staff model mixes staff roles: a first contact by a care coordinator or pharmacy technician marks a patient still in the outreach phase, while a first contact by a community health worker marks a patient already handed to a field team, so an "assigned staff" effect across roles partly measures the patient's path rather than the staff member. Two specifications are therefore reported. (1) All staff, with role added to the fixed effects (market by quarter by role), so that staff are compared only with staff in the same role, market, and quarter. (2) Community health workers only: the assigned staff member is the community health worker who made the patient's first completed contact within 30 days of activation, among patients who had such a contact, with market by quarter fixed effects; this is the comparison a program can act on (which community health worker a patient is assigned to). Validation, balance, heterogeneity, and profile associations are reported for both, and the balance gate applies to each.
