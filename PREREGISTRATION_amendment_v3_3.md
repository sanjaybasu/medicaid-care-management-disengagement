# Pre-registration amendment v3.3: staff-assignment instruments for care-team actions

Dated 2026-09-10, written after inspecting the distribution of staff-level action rates and before any instrument-outcome result was examined. Amends `PREREGISTRATION_v3_disengagement.md` (Section 7) and its amendments. Rules are fixed here; results are written only by the pipeline to `results/canonical.json`.

## I1. Question

Which care-team actions in a patient's first 30 days reduce disengagement, and for whom, when the action is not chosen by the patient's own circumstances? Actions: (a) any in-person contact in days 0 to 30 after activation; (b) a second completed contact within 7 days of the first; (c) any weekend contact in days 0 to 30. Outcome: disengagement in days 31 to 120 after activation, defined as no completed contact in that window and no program completion by day 120, among patients enrolled in the health plan for all of days 31 to 120.

## I2. Design

Patients are assigned to a care team member for their first contact by caseload and geography, not by their engagement prospects. Staff members differ in how often they take each action. The instrument for each action is the assigned first-contact staff member's leave-one-patient-out rate of that action among their other patients (the "inferred preference"), computed only from the log; no staff member is surveyed. The design is the preference-based instrument of the pharmacoepidemiology and judge-assignment literatures. One row per patient; the unit of assignment is the staff member.

## I3. Estimation

Two-stage least squares with market by activation-quarter fixed effects and baseline covariates (age, gender, race and ethnicity, risk percentile, condition flags, prior acute care events, days from referral to activation where available), standard errors clustered by staff member. Reported: first-stage coefficient and effective F; reduced form; the local average treatment effect as a risk difference with 95% CI; the ordinary least squares association for comparison. Heterogeneity: the instrumented effect within pre-specified subgroups (state; age under 45 versus 45 and over; behavioral health diagnosis; risk percentile above versus below the median; prior acute care event in 365 days), and a doubly robust instrumental-variable learner (DRIV) for the conditional effect, reported as effects by quartile of the learned score on held-out folds.

## I4. Falsification gates, fixed before results

Gate 1 (relevance): effective first-stage F of at least 10. Gate 2 (balance): no baseline covariate differs across the instrument by more than 0.10 standard deviations per standard deviation of the instrument after fixed effects, and the joint test of all covariates on the instrument has p above 0.05. Gate 3 (monotonicity): the first stage has the same sign in every market with at least 200 patients. Gate 4 (exclusion): the estimate changes by less than half its standard error when the assigned staff member's other two action rates are added as controls, and staff role is controlled. An action whose instrument fails Gate 1 or Gate 2 is reported as not estimable by this design. An action that fails Gate 3 or Gate 4 is reported with the failure stated and is not interpreted causally.

## I5. Reporting

A single supplementary table with, per action, the first stage, balance summary, gate results, OLS and 2SLS estimates, and subgroup estimates; a main-text paragraph only if an action passes all four gates. Results that fail the gates are reported as such, because a null or unidentifiable answer bears on whether programs should invest in the action.
