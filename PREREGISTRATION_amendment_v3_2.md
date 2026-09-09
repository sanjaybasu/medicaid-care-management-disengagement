# Pre-registration amendment v3.2: eligibility, temporal split, physician review of clinical stakes, and reporting changes

Dated 2026-09-09, written before the re-run results were examined. Amends `PREREGISTRATION_v3_disengagement.md` and `PREREGISTRATION_amendment_v3_1.md`. Rules are fixed here; results are written only by the pipeline to `results/canonical.json`.

## H1. Eligibility

A decision point is eligible if the patient was enrolled in the health plan for all 90 of the 90 days after the contact (previously at least 60 of 90). The relaxed 60-day rule and the no-restriction specification move to the sensitivity grid (Supplementary Table S3). The lever analyses require enrollment for all 90 days of days 31 to 120.

## H2. Temporal split

Training uses contacts of training-era patients (activated through 2025-06-30) dated before 2025-07-01 only, so that no training contact or note is contemporaneous with any test-era contact. Test-era patients are those activated on or after 2025-07-01; their contacts run to the last date with complete 90-day follow-up. Contacts of training-era patients dated on or after 2025-07-01 are excluded from the primary analyses and counted in the flow diagram. The rolling-landmark and quarterly-refit analyses (v3.1 Sections A and G), which split by date rather than by cohort, use the enrollment criterion alone.

## H3. Blinded physician review of clinical stakes (replaces v3.1 Section B)

Purpose. Establish whether patients who disengage leave with clinical work unfinished, compared with patients who complete the program. The review does not re-define engagement or disengagement, which are measured objectively from the contact log, and it does not ask reviewers to predict the outcome.

Sample. From test-era eligible decision points, the last completed contact before disengagement for 100 patients whose contact was followed by no completed contact in 90 days, and the last completed contact before program completion for 100 patients whose contact was followed by a completion status within 90 days; one index contact per patient; random selection with seed 20260909. Packets contain the structured summary as of the index contact, the notes of the prior 90 days, and the index note, with nothing after the index contact, no model score, and no group label.

Reviewers and forms. Three physicians. Each case is read by one physician; 20% of cases in each group are read by a second physician for agreement. Two items per case: (1) open clinical need at the index contact (medication gap, pending referral or appointment, uncontrolled condition, active social crisis, behavioral health need, none), multiple allowed; (2) harm if the patient had no further contact for 90 days (low, moderate, high). Free-text comments.

Power. With 100 cases per group, a two-sided test at alpha 0.05 has 80% power to detect a difference of 15 percentage points in the share with any open clinical need (0.75 against 0.90). This is the smallest sample that answers the question, chosen to limit reviewer burden to about 80 cases per physician.

Analysis. Share of cases with each need, with any need, and with harm rated high, by group, with case-bootstrap intervals and the between-group difference; Cohen's kappa and percent agreement on the double-read cases. Forms are completed by the physicians themselves; no form is pre-filled by any procedure.

## H4. Reporting changes

Sensitivity and specificity accompany every operating-point result. A confusion matrix at the 20% flagging threshold is reported in the main text. The Youden-index operating point is reported alongside the capacity-based threshold (Supplementary Table S19). Equalized odds (true and false positive rates by race and ethnicity and by state at the single overall threshold, with the ratio of the lowest to the highest group value) are reported (Supplementary Table S20). Brier scores carry bootstrap intervals. The in-person contact lever (v3 Section 7) is reported in the Supplement only, alongside the v3.1 Section D levers; the three exposures without equipoise support are omitted from tables and named in the legend. The acute care comparison after disengagement is reported in the Supplement only.
