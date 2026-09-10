#!/bin/zsh
# Re-run the analysis chain after the eligibility (90 of 90 enrolled days) and temporal-split (training contacts before 2025-07-01) changes.
set -e; cd /Users/sanjaybasu/waymark-local/packaging/care-management-disengagement; PY=/opt/anaconda3/bin/python3
run() { echo "=== $1 $(date '+%H:%M:%S')"; $PY scripts/$1 ${2:-} > results/${3}_log.txt 2>&1; }
run 01_outcomes.py "" outcomes
run 04_models.py --with-embeddings models_emb
run 05_subgroups.py "" subgroups
run 06_levers.py "" levers
run 07_sequence.py "" sequence
run 09_sensitivity.py "" sensitivity
run 10_landmark.py "" landmark
run 13_simple_score.py "" simple_score
run 14_levers2.py "" levers2
run 15_capacity.py "" capacity
run 16_fairness.py "" fairness
run 17_clinical_stakes.py "" clinical_stakes
run 18_metric_cis.py "" metric_cis
run 19_ablation.py "" ablation
run 20_rolling_refit.py "" rolling_refit
run 22_table_cis.py "" table_cis
run 23_staff_preference_iv.py "" staff_iv
run 24_staff_effects.py "" staff_effects
run 25_attempts.py "" attempts
echo "=== PIPELINE DONE $(date '+%H:%M:%S')"
