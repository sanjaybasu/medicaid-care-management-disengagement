"""Step 19: feature-group ablation of the full model (leave one group out) and a logistic-regression learner on the same features, test era.
Uses the design matrices saved by 04_models.py (X5_tr.npy, X5_te.npy) whose column blocks are: structured (+ categorical), contact history
(+ categorical), lexicon and tags, note embeddings (128), TF-IDF stacks (2). Writes results/ablation.json."""
import json, pathlib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import sys; sys.path.insert(0, 'scripts'); from _boot import boot_ci
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); HP = json.load(open(R/"hyperparameters.json"))["gbm"]
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet"); df = F.merge(O[["decision_id","eligible","y_primary"]], on="decision_id"); df = df[df.eligible == 1].reset_index(drop=True)
STRUCT = ["age","risk_percentile","tier1_flg","days_from_zd","days_since_last_adt"] + [f"adt_all_prior_{w}d" for w in [30,90,180,365]] + [c for c in ["diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip","no_pcp_last_10mo"] if c in df.columns]
HIST = ["n_prior","n_prior_90","n_prior_30","days_since_last_contact","chw90","pharm90","cpht90","ther90","cc90","inp90","textmod90","n_disc90","gap_mean","gap_max","cur_inp","cur_textmod","note_len","goals_open","goals_completed_before","goals_social","goals_clinical","goals_bh"]
LEX = [c for c in df.columns if c.startswith("lex_")]; TAGS = [c for c in df.columns if c.startswith("tag_")]
n_struct = len(STRUCT) + pd.get_dummies(df[["gender","race","state"]].fillna("u"), drop_first=True).shape[1]; n_hist = len(HIST) + pd.get_dummies(df[["cur_rg","prev_rg","contact_type","encounter_type"]].fillna("u"), drop_first=True).shape[1]; n_lt = len(LEX) + len(TAGS)
Xtr, Xte = np.load(D/"X5_tr.npy"), np.load(D/"X5_te.npy"); tr = (df.training_era == 1).values; y = df.y_primary.values; ytr, yte = y[tr], y[~tr]
assert Xtr.shape[1] == n_struct + n_hist + n_lt + 128 + 2, (Xtr.shape, n_struct, n_hist, n_lt)
blocks = {"structured": np.arange(0, n_struct), "contact_history": np.arange(n_struct, n_struct+n_hist), "lexicon_and_tags": np.arange(n_struct+n_hist, n_struct+n_hist+n_lt), "note_embeddings": np.arange(n_struct+n_hist+n_lt, n_struct+n_hist+n_lt+128), "tfidf_note_stack": np.arange(Xtr.shape[1]-2, Xtr.shape[1])}
IDS = df.person_id.values[~tr]
def ev(a, b):
    c = boot_ci(yte, b, IDS, {"auroc": roc_auc_score, "auprc": average_precision_score}, B=300); return {"auroc": round(float(roc_auc_score(yte, b)), 4), "auprc": round(float(average_precision_score(yte, b)), 4), "auroc_ci_95": c["auroc"]["ci_95"], "auprc_ci_95": c["auprc"]["ci_95"]}
full = HistGradientBoostingClassifier(**HP).fit(Xtr, ytr).predict_proba(Xte)[:,1]; out = {"full_model": ev(None, full), "leave_one_group_out": {}, "only_one_group": {}}
for g, cols in blocks.items():
    keep = np.setdiff1d(np.arange(Xtr.shape[1]), cols); p = HistGradientBoostingClassifier(**HP).fit(Xtr[:, keep], ytr).predict_proba(Xte[:, keep])[:,1]; r = ev(None, p); r["auprc_drop"] = round(out["full_model"]["auprc"] - r["auprc"], 4); r["auroc_drop"] = round(out["full_model"]["auroc"] - r["auroc"], 4); out["leave_one_group_out"][g] = r; print("without", g, r, flush=True)
    p1 = HistGradientBoostingClassifier(**HP).fit(Xtr[:, cols], ytr).predict_proba(Xte[:, cols])[:,1]; out["only_one_group"][g] = ev(None, p1)
sc = StandardScaler().fit(Xtr); lr = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(Xtr), ytr); out["logistic_regression_same_features"] = ev(None, lr.predict_proba(sc.transform(Xte))[:,1])
for hp_name, hp in [("shallower_trees_max_leaf_nodes_7", {**HP, "max_leaf_nodes": 7}), ("deeper_trees_max_leaf_nodes_31", {**HP, "max_leaf_nodes": 31}), ("learning_rate_0.1", {**HP, "learning_rate": 0.1})]:
    out.setdefault("hyperparameter_variants", {})[hp_name] = ev(None, HistGradientBoostingClassifier(**hp).fit(Xtr, ytr).predict_proba(Xte)[:,1])
json.dump(out, open(R/"ablation.json", "w"), indent=1); print(json.dumps(out, indent=1))
