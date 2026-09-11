# Pre-registration amendment v3.7: restriction to rising-risk (tier 1) patients

Dated 2026-09-10, written before the re-run results were examined. Rationale: patients activated for a quality-measure (HEDIS) need alone may require a single contact to schedule an appointment, after which no further contact is expected; counting the absence of further contact as disengagement misdescribes their episode. The program's rising-risk (tier 1) flag identifies the patients for whom continued engagement is the goal.

## M1. Change

Eligibility for every analysis (primary models, subgroups, leave-one-state-out, frozen-model and quarterly-refit validation, staff-assignment and staff-effect analyses, attempt experiments, operational rule, deployment export) requires the tier 1 flag at activation in addition to the 90-day enrollment rule and the temporal split. Patients without the flag are counted in the flow diagram as excluded. All other definitions, models, contrasts, gates, and thresholds are unchanged.

## M2. Reporting

The flow diagram, Table 1, and the cohort paragraph report the number excluded; the sensitivity grid (Supplementary Table S3) gains a row with the tier 1 restriction removed so that the effect of the restriction on discrimination is visible.
