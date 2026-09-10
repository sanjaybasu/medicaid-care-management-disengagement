# Pre-registration amendment v3.6: an operational "worth attempting" rule

Dated 2026-09-10, written before the results were examined. Purpose: give care teams a rule that is simpler than a randomized protocol and that they can judge from its own error rates.

## L1. The rule

At day 7 after a completed contact, flag the patient if all three hold: (a) the contact-history model (M2, no note text) placed the contact in the top 20% of predicted disengagement risk; (b) no completed contact and no logged attempt occurred in days 1 to 7 (lapsed and unattended); (c) the predicted return if attempted is above the median of lapsed patients, where predicted return is the difference between the modeled probability of a completed contact in days 8 to 30 with and without an attempt in days 1 to 7 (a T-learner: separate gradient-boosted outcome models in attempted and unattempted lapsed episodes, using age, risk percentile, days since activation, contacts in the prior 90 days, attempts in the prior 30 days, the patient's prior attempt connect rate, state, and channel of the index contact).

## L2. Evaluation, fixed before results

Rule components (a) and (b) are evaluated on the test set against disengagement within 90 days: confusion matrix, sensitivity, specificity, positive and negative predictive value, and the share of contacts flagged, alongside M2 alone at 20% flagged and the hand-tallied points score. Component (c) is evaluated by split sample on all lapsed episodes: the uplift model is fit on a random half, the other half is ranked by predicted uplift, and the doubly robust effect of attempting on return is compared between the half ranked above and below the median; the gate is a difference with a 95% bootstrap interval excluding zero. Observed return with and without an attempt is reported by uplift tertile in the held-out half, together with attempts needed per additional returned patient (1 divided by the effect) in each tertile. Equity: the share of lapsed patients screened out by component (c) is reported by race and ethnicity and by state.

## L3. Reporting

A supplementary table (S25) and the operations memo. If component (c) fails its gate, the memo recommends the two-part rule (a) and (b) alone and states that no futility screen is supported.
