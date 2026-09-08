"""Step 9: sensitivity grid (PREREGISTRATION_v3 Table 5). Re-fits M2 (structured + history) and M3 (M2 + TF-IDF note stack) under alternative
specifications and evaluates on the test era. Writes results/sensitivity.json. Embedding features are not used here (M3 is the text model)."""
import json, pathlib, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import sys; sys.path.insert(0, 'scripts'); from _boot import boot_ci, std_fns
import numpy as np
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); HP = json.load(open(R/"hyperparameters.json"))["gbm"]
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet")
df = F.merge(O.drop(columns=["person_id","enc_date","training_era","state"]), on="decision_id"); df["enc_date"] = pd.to_datetime(df.enc_date)
STRUCT = ["age","risk_percentile","tier1_flg","days_from_zd","days_since_last_adt"] + [f"adt_all_prior_{w}d" for w in [30,90,180,365]] + ["diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip","no_pcp_last_10mo"]
HIST = ["n_prior","n_prior_90","n_prior_30","days_since_last_contact","chw90","pharm90","cpht90","ther90","cc90","inp90","textmod90","n_disc90","gap_mean","gap_max","cur_inp","cur_textmod","note_len","goals_open","goals_completed_before","goals_social","goals_clinical","goals_bh"]
def design(d, extra=()):
    return pd.concat([d[STRUCT + HIST + list(extra)].astype(float), pd.get_dummies(d[["gender","race","cur_rg","prev_rg","contact_type","encounter_type"]].fillna("u"), drop_first=True, dtype=float)], axis=1).fillna(0)
def run(d, tr, te, ycol, extra=()):
    X = design(d, extra).values; y = d[ycol].values; txt = d.note_text.fillna("").values
    p2 = HistGradientBoostingClassifier(**HP).fit(X[tr], y[tr]).predict_proba(X[te])[:,1]
    vec = TfidfVectorizer(min_df=20, max_df=0.5, ngram_range=(1,2), sublinear_tf=True, max_features=50000); Xt = vec.fit_transform(txt[tr]); Xte = vec.transform(txt[te]); oof = np.zeros(tr.sum())
    for a, b in GroupKFold(5).split(Xt, y[tr], d.person_id.values[tr]): oof[b] = LogisticRegression(C=0.5, max_iter=3000).fit(Xt[a], y[tr][a]).predict_proba(Xt[b])[:,1]
    pte = LogisticRegression(C=0.5, max_iter=3000).fit(Xt, y[tr]).predict_proba(Xte)[:,1]
    p3 = HistGradientBoostingClassifier(**HP).fit(np.column_stack([X[tr], oof]), y[tr]).predict_proba(np.column_stack([X[te], pte]))[:,1]
    ids = d.person_id.values[te]; fns = {"auroc": roc_auc_score, "auprc": average_precision_score}
    m2 = boot_ci(y[te], p2, ids, fns); m3 = boot_ci(y[te], p3, ids, fns)
    # gain interval: paired bootstrap of the difference
    rng = np.random.default_rng(20260908); up = np.unique(ids); idx = {q: np.where(ids == q)[0] for q in up}; g = []
    for _ in range(300):
        ix = np.concatenate([idx[q] for q in rng.choice(up, len(up))]); yy = y[te][ix]
        if yy.min() != yy.max(): g.append(average_precision_score(yy, p3[ix]) - average_precision_score(yy, p2[ix]))
    return {"n_train": int(tr.sum()), "n_test": int(te.sum()), "test_event_rate": round(float(y[te].mean()), 4), "M2_structured_history": {k: v["estimate"] for k, v in m2.items()}, "M3_plus_tfidf_text": {k: v["estimate"] for k, v in m3.items()}, "ci": {"M2_structured_history": m2, "M3_plus_tfidf_text": m3}, "auprc_gain_text": round(float(average_precision_score(y[te], p3) - average_precision_score(y[te], p2)), 4), "auprc_gain_text_ci_95": [round(float(np.percentile(g, 2.5)), 4), round(float(np.percentile(g, 97.5)), 4)]}
out = {}
base = df[df.eligible == 1].reset_index(drop=True); tr = (base.training_era == 1).values
out["base_eligible60_activation_split"] = run(base, tr, ~tr, "y_primary")
e90 = df[df.enrolled_days_90 >= 90].reset_index(drop=True); tr90 = (e90.training_era == 1).values
out["eligibility_90_enrolled_days"] = run(e90, tr90, ~tr90, "y_primary")
alld = df.reset_index(drop=True); tra = (alld.training_era == 1).values
out["no_eligibility_restriction_enrollment_covariate"] = run(alld, tra, ~tra, "y_primary", extra=("enrolled_days_90",))
cut = pd.Timestamp("2025-07-01"); trc = ((base.training_era == 1) & (base.enc_date < cut)).values; tec = (base.training_era == 0).values
out["training_contacts_before_2025_07_01_only"] = run(base, trc, tec, "y_primary")
for yc in ["y_30", "y_silent", "y_explicit"]: out[f"outcome_{yc}"] = run(base, tr, ~tr, yc)
json.dump(out, open(R/"sensitivity.json", "w"), indent=1); print(json.dumps(out, indent=1))
