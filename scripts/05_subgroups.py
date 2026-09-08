"""Step 5: subgroup, leave-one-state-out, and interpretability (Section 6). Writes results/subgroups.json"""
import json, pathlib, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); HP = json.load(open(R/"hyperparameters.json"))["gbm"]
P = pd.read_parquet(D/"test_predictions.parquet"); full = "M5_full" if "M5_full" in P.columns else "M3b_plus_lexicon_tags"
out = {"full_model": full, "by_state": {}, "by_race": {}, "by_days_since_activation": {}, "by_activating_discipline": {}}
def m(y, p): return {"n": int(len(y)), "events": int(y.sum()), "auroc": round(float(roc_auc_score(y, p)), 3), "auprc": round(float(average_precision_score(y, p)), 3)} if len(np.unique(y)) > 1 and len(y) >= 200 else {"n": int(len(y)), "note": "insufficient"}
def both(mask):
    return {"structured_history": m(P.y[mask].values, P.M2_structured_history[mask].values), "full": m(P.y[mask].values, P[full][mask].values),
            "auroc_gain": round(float(roc_auc_score(P.y[mask], P[full][mask]) - roc_auc_score(P.y[mask], P.M2_structured_history[mask])), 4) if len(np.unique(P.y[mask])) > 1 and mask.sum() >= 200 else None}
for st in ["VIRGINIA","WASHINGTON","OHIO"]: out["by_state"][st] = both((P.state == st).values)
for rc in ["Black or African American","White","Hispanic","Asian"]:
    mk = (P.race == rc).values
    if mk.sum() >= 200: out["by_race"][rc] = both(mk)
for lab, lo, hi in [("0_30", -1, 30), ("31_90", 30, 90), ("91_180", 90, 180), ("181_365", 180, 366)]: out["by_days_since_activation"][lab] = both(((P.days_from_zd > lo) & (P.days_from_zd <= hi)).values)
for rg in ["CHW","CC","CPhT","PharmD","Therapist"]:
    mk = (P.cur_rg == rg).values
    if mk.sum() >= 200: out["by_activating_discipline"][rg] = both(mk)
# leave-one-state-out on the full analytic set (train two states, test third), structured+history vs +text stack
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet"); df = F.merge(O[["decision_id","eligible","y_primary"]], on="decision_id"); df = df[df.eligible == 1].reset_index(drop=True)
STRUCT = ["age","risk_percentile","tier1_flg","days_from_zd","days_since_last_adt"] + [f"adt_all_prior_{w}d" for w in [30,90,180,365]] + [c for c in ["diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip","no_pcp_last_10mo"] if c in df.columns]
HIST = ["n_prior","n_prior_90","n_prior_30","days_since_last_contact","chw90","pharm90","cpht90","ther90","cc90","inp90","textmod90","n_disc90","gap_mean","gap_max","cur_inp","cur_textmod","note_len","goals_open","goals_completed_before","goals_social","goals_clinical","goals_bh"]
LEX = [c for c in df.columns if c.startswith("lex_")]; TAGS = [c for c in df.columns if c.startswith("tag_")]
X = pd.concat([df[STRUCT + HIST + LEX + TAGS].astype(float), pd.get_dummies(df[["gender","race","cur_rg","prev_rg","contact_type","encounter_type"]].fillna("u"), drop_first=True, dtype=float)], axis=1).fillna(0)
Xh = X[[c for c in X.columns if not (c.startswith("lex_") or c.startswith("tag_"))]]
y = df.y_primary.values; txt = df.note_text.fillna("").values
out["leave_one_state_out"] = {}
for st in ["VIRGINIA","WASHINGTON","OHIO"]:
    tr = (df.state != st).values; te = ~tr
    p2 = HistGradientBoostingClassifier(**HP).fit(Xh.values[tr], y[tr]).predict_proba(Xh.values[te])[:,1]
    vec = TfidfVectorizer(min_df=20, max_df=0.5, ngram_range=(1,2), sublinear_tf=True, max_features=50000); Xt = vec.fit_transform(txt[tr]); Xte = vec.transform(txt[te])
    oof = np.zeros(tr.sum())
    for a, b in GroupKFold(5).split(Xt, y[tr], df.person_id[tr]): oof[b] = LogisticRegression(C=0.5, max_iter=3000).fit(Xt[a], y[tr][a]).predict_proba(Xt[b])[:,1]
    pte = LogisticRegression(C=0.5, max_iter=3000).fit(Xt, y[tr]).predict_proba(Xte)[:,1]
    pf = HistGradientBoostingClassifier(**HP).fit(np.column_stack([X.values[tr], oof]), y[tr]).predict_proba(np.column_stack([X.values[te], pte]))[:,1]
    out["leave_one_state_out"][f"test_{st}"] = {"n": int(te.sum()), "event_rate": round(float(y[te].mean()), 3), "structured_history": m(y[te], p2), "full_text": m(y[te], pf)}
    print(st, out["leave_one_state_out"][f"test_{st}"], flush=True)
# interpretability: permutation importance of the full (non-embedding) model on the temporal test era
tr = (df.training_era == 1).values; te = ~tr
mdl = HistGradientBoostingClassifier(**HP).fit(X.values[tr], y[tr]); sub = np.random.default_rng(1).choice(np.where(te)[0], 6000, replace=False)
pi = permutation_importance(mdl, X.values[sub], y[sub], scoring="average_precision", n_repeats=5, random_state=1, n_jobs=4)
imp = sorted(zip(X.columns, pi.importances_mean), key=lambda t: -t[1])[:25]; out["permutation_importance_top25_auprc"] = [{"feature": f, "drop_in_auprc": round(float(v), 4)} for f, v in imp]
# lexicon univariate: disengagement rate when flag present vs absent (test era)
out["lexicon_rates_test_era"] = {k: {"prevalence": round(float(df[k+"_cur"][te].mean()), 4), "rate_if_present": round(float(y[te][df[k+"_cur"].values[te] == 1].mean()), 3) if (df[k+"_cur"].values[te] == 1).sum() >= 50 else None, "rate_if_absent": round(float(y[te][df[k+"_cur"].values[te] == 0].mean()), 3)} for k in sorted(set(c[:-4] for c in LEX if c.endswith("_cur")))}
json.dump(out, open(R/"subgroups.json", "w"), indent=1); print(json.dumps({k: out[k] for k in ["by_state","by_race","by_days_since_activation"]}, indent=1)[:3000]); print(out["permutation_importance_top25_auprc"][:12]); print(out["lexicon_rates_test_era"])
