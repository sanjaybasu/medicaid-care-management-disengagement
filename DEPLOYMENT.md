# Deployment guide: disengagement flag and day-7 work list

For data science, product, engineering, and analytics. Everything here is derived from the pre-registered analysis in this repository (see `README.md` and the `PREREGISTRATION_*` documents); no patient data are in the repository. Artifacts are in `deploy/`.

## What the flag is

At each completed care-team contact, the contact-history model (`deploy/m2_*.joblib`) predicts the probability that the patient will have no completed contact and no program completion in the next 90 days. Contacts in the top 20% of predicted probability are flagged. On patients activated from July 2025 onward, whom the model never saw in training, flagging 20% of contacts caught 63% of disengagements with 60 true disengagements per 100 flags, against 22 per 100 for the acute-care risk percentile that currently orders follow-up (which is at chance for this outcome). Full error rates, including the confusion matrix, are in `deploy/day7_rule.json` and in the paper's Table 4.

## Two ways to compute it

1. **Model.** Build the 62 feature columns listed in `deploy/feature_spec.json` (`columns_in_order`), in that order, for each completed contact, and call `predict_proba` on the joblib model. Use `m2_deploy_refit_all_data_through_2026-06-07.joblib` for production; `m2_validated_trained_before_2025-07-01.joblib` is the exact model whose performance the paper reports. Feature definitions are in the same file, and `scripts/02_features.py` is the reference implementation against the lighthouse extracts (EncounterNote for contacts and attempts, status history, goals, ADT events, Signal risk percentile and tier, condition flags at activation). One-hot columns use `drop_first` with missing values filled as `u`; only the listed columns are used.
2. **Points score.** If a model call is not practical, `deploy/points_score.json` gives an eight-item integer score a coordinator can tally by hand (contacts in the prior 90 days, days since last contact, first 30 days, outreach encounter, coordinator or technician contact, "declined" in the note, positive-engagement language in the note, open goals). It captures about 80% of the model's gain over the risk percentile.

## The day-7 work list

`deploy/day7_rule.json` specifies the list a team works from. Seven days after a completed contact, a patient is listed if (a) the contact was flagged and (b) there has been no completed contact and no logged attempt since (an attempt is an EncounterNote with `encounterOccurred = 'NO'`). On the test set this list held 13% of contacts and 80 of every 100 patients on it went on to disengage. Order the list by predicted return if attempted, from the two uplift models in `deploy/uplift_*.joblib` with the columns in `deploy/uplift_spec.json` (uplift = prediction of the attempted model minus the not-attempted model); call the highest first. Do not use the uplift to exclude anyone: a hard screen would drop Hispanic and Washington patients more often than others.

## Thresholds and refitting

`deploy/thresholds.json` gives the probability cutpoint for a 20% flag rate and the decile cutpoints. Disengagement rates fall over time, so recompute the cutpoint on the most recent completed quarter and refit the model quarterly (`scripts/20_rolling_refit.py` shows the effect of refitting). Report monthly: realized disengagement by predicted decile, flag rate and sensitivity by state and by race and ethnicity, and attempts logged per listed patient.

## What the flag does not do

It does not say what to do for a flagged patient. The pre-registered within-patient analyses (Supplementary Table S24 of the paper) indicate that an attempt in the week after a lapse, made by phone on a weekday morning, is the best-supported next step, and that an early second contact helps most for younger patients who have not yet used acute care. Whether working the list reduces disengagement is untested; the recommended test is to start the list in a random half of teams first.

## Contact

Sanjay Basu (sanjay.basu@waymarkcare.com). Code and pre-registration: this repository; the public mirror is github.com/sanjaybasu/medicaid-care-management-disengagement.
