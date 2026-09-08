"""Step 13 (amendment v3.1 Section C): eight-predictor logistic model and integer points score, trained on the training era, evaluated on the test era.
Writes results/simple_score.json."""
import json, pathlib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import sys; sys.path.insert(0, 'scripts'); from _boot import boot_ci
D = pathlib.Path("data_cache"); R = pathlib.Path("results")
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet"); df = F.merge(O[["decision_id","eligible","y_primary"]], on="decision_id"); df = df[df.eligible == 1].reset_index(drop=True)
P = pd.read_parquet(D/"test_predictions.parquet").set_index("decision_id")
X = pd.DataFrame({"prior_contacts_90d_log": np.log1p(df.n_prior_90), "days_since_last_contact_log": np.log1p(df.days_since_last_contact.fillna(365)), "first_30_days": (df.days_from_zd <= 30).astype(float), "encounter_patient_outreach": (df.encounter_type == "PATIENT_OUTREACH").astype(float), "contact_by_coordinator_or_technician": df.cur_rg.isin(["CC","CPhT"]).astype(float), "note_declined": df.lex_declined_cur.astype(float), "note_positive_engagement": df.lex_positive_engagement_cur.astype(float), "open_goals_log": np.log1p(df.goals_open.fillna(0))})
y = df.y_primary.values; tr = (df.training_era == 1).values; te = ~tr
lr = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr]); p = lr.predict_proba(X[te])[:,1]
coef = dict(zip(X.columns, lr.coef_[0])); unit = max(abs(v) for v in coef.values())/10; pts = {k: int(round(v/unit)) for k, v in coef.items()}  # largest coefficient = 10 points
score = (X[te] * pd.Series(pts)).sum(1).values
def nb20(yy, s):
    thr = np.quantile(s, 0.8); f = s >= thr; return {"per_100_flags": round(float(100*yy[f].mean()), 1), "sensitivity": round(float(yy[f].sum()/yy.sum()), 3), "flag_rate": round(float(f.mean()), 3)}
lp = np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))); cal = LogisticRegression(C=1e6).fit(lp.reshape(-1,1), y[te])
out = {"predictors": list(X.columns), "coefficients": {k: round(float(v), 3) for k, v in coef.items()}, "intercept": round(float(lr.intercept_[0]), 3), "points": pts, "n_test": int(te.sum()),
       "logistic": {"auroc": round(float(roc_auc_score(y[te], p)), 4), "auprc": round(float(average_precision_score(y[te], p)), 4), "brier": round(float(brier_score_loss(y[te], p)), 4), "calibration_slope": round(float(cal.coef_[0][0]), 3), "at_20pct_flagged": nb20(y[te], p)},
       "points_score": {"auroc": round(float(roc_auc_score(y[te], score)), 4), "auprc": round(float(average_precision_score(y[te], score)), 4), "at_20pct_flagged": nb20(y[te], score), "score_range": [int(score.min()), int(score.max())]}}
Pt = P.loc[df.decision_id[te]]; m0, m5 = average_precision_score(Pt.y, Pt.M0_signal_risk), average_precision_score(Pt.y, Pt.M5_full)
out["share_of_full_model_gain_over_M0_captured"] = {"logistic": round(float((out["logistic"]["auprc"] - m0)/(m5 - m0)), 3), "points_score": round(float((out["points_score"]["auprc"] - m0)/(m5 - m0)), 3)}
ids = df.person_id.values[te]
def fl(yy, pp): thr = np.quantile(pp, 0.8); f = pp >= thr; return {"per_100_flags": 100*yy[f].mean(), "sensitivity": yy[f].sum()/max(yy.sum(),1)}
fns = {"auroc": roc_auc_score, "auprc": average_precision_score, "per_100_flags": lambda yy, pp: fl(yy, pp)["per_100_flags"], "sensitivity": lambda yy, pp: fl(yy, pp)["sensitivity"]}
out["ci"] = {"logistic": boot_ci(y[te], p, ids, fns), "points_score": boot_ci(y[te], score.astype(float), ids, fns)}
json.dump(out, open(R/"simple_score.json", "w"), indent=1); print(json.dumps(out, indent=1))
