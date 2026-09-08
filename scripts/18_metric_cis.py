"""Step 18: full metric set with 95% member-bootstrap intervals for every model on the test era, at the 20% flag threshold (80th percentile of each
model's test-era predictions); decision-point-level bootstrap for the full model as a comparison of interval width; subgroup AUROC intervals.
Reads data_cache/test_predictions.parquet only. Writes results/metric_cis.json."""
import json, pathlib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); P = pd.read_parquet(D/"test_predictions.parquet"); y = P.y.values.astype(int); B = 1000; rng = np.random.default_rng(20260908)
MODELS = [c for c in ["M0_signal_risk","M1_structured","M2_structured_history","M3_plus_tfidf_text","M3b_plus_lexicon_tags","M4_plus_embeddings","M5_full","text_only_tfidf","embeddings_only","seq_GRU","seq_Transformer","seq_Mamba_SSM"] if c in P.columns]
def calib(yy, p):
    lp = np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))); lr = LogisticRegression(C=1e6, max_iter=500).fit(lp.reshape(-1,1), yy); return float(lr.coef_[0][0]), float(lr.intercept_[0])
def allm(yy, p, thr):
    f = p >= thr; tp = int((f & (yy==1)).sum()); fp = int((f & (yy==0)).sum()); fn = int((~f & (yy==1)).sum()); tn = int((~f & (yy==0)).sum()); n = len(yy)
    sens = tp/max(tp+fn,1); spec = tn/max(tn+fp,1); ppv = tp/max(tp+fp,1); npv = tn/max(tn+fn,1); f1 = 2*tp/max(2*tp+fp+fn,1); nb = tp/n - fp/n*(thr/(1-thr)) if thr < 1 else 0
    sl, ic = calib(yy, p)
    return {"auroc": roc_auc_score(yy, p), "auprc": average_precision_score(yy, p), "brier": brier_score_loss(yy, p), "calibration_slope": sl, "calibration_intercept": ic, "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv, "f1": f1, "net_benefit": nb, "flag_rate": f.mean(), "per_100_flags": 100*ppv}
pids = P.person_id.values; up = np.unique(pids); idx = {p: np.where(pids == p)[0] for p in up}
boots = [np.concatenate([idx[p] for p in rng.choice(up, len(up))]) for _ in range(B)]; boots_dp = [rng.integers(0, len(y), len(y)) for _ in range(B)]
out = {"n_test": int(len(y)), "n_members": int(len(up)), "events": int(y.sum()), "bootstrap": "1,000 resamples of members; percentile intervals", "threshold_rule": "80th percentile of each model's test-era predictions (20% of contacts flagged)", "models": {}}
for m in MODELS:
    p = P[m].values; thr = float(np.quantile(p, 0.8)); pt = allm(y, p, thr); bs = {k: [] for k in pt}
    for ix in boots:
        r = allm(y[ix], p[ix], thr); [bs[k].append(v) for k, v in r.items()]
    out["models"][m] = {k: {"estimate": round(float(v), 4), "ci_95": [round(float(np.percentile(bs[k], 2.5)), 4), round(float(np.percentile(bs[k], 97.5)), 4)]} for k, v in pt.items()}; out["models"][m]["threshold"] = round(thr, 4); print(m, {k: (out['models'][m][k]['estimate'], out['models'][m][k]['ci_95']) for k in ['auroc','auprc','sensitivity','specificity','f1']}, flush=True)
# decision-point bootstrap for the full model, to show interval width difference
p = P.M5_full.values; thr = float(np.quantile(p, 0.8)); bsd = {"auroc": [], "auprc": []}
for ix in boots_dp: bsd["auroc"].append(roc_auc_score(y[ix], p[ix])); bsd["auprc"].append(average_precision_score(y[ix], p[ix]))
out["M5_full_decision_point_bootstrap"] = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in bsd.items()}
# subgroup AUROC and AUPRC intervals (500 member resamples within subgroup) for M2 and M5
out["subgroups"] = {}
for dim in ["state", "race"]:
    for lev, g in P.groupby(dim):
        if len(g) < 200 or g.y.nunique() < 2 or (dim == "race" and lev not in ["Black or African American","White","Hispanic","Asian"]): continue
        gp = g.person_id.values; gu = np.unique(gp); gi = {q: np.where(gp == q)[0] for q in gu}; rec = {"n": int(len(g))}
        for m in ["M2_structured_history","M5_full"]:
            a, pr = [], []
            for _ in range(500):
                ix = np.concatenate([gi[q] for q in rng.choice(gu, len(gu))]); yy = g.y.values[ix]
                if yy.min() == yy.max(): continue
                a.append(roc_auc_score(yy, g[m].values[ix])); pr.append(average_precision_score(yy, g[m].values[ix]))
            rec[m] = {"auroc": round(float(roc_auc_score(g.y, g[m])), 3), "auroc_ci_95": [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)], "auprc": round(float(average_precision_score(g.y, g[m])), 3), "auprc_ci_95": [round(float(np.percentile(pr, 2.5)), 3), round(float(np.percentile(pr, 97.5)), 3)]}
        out["subgroups"][f"{dim}:{lev}"] = rec
json.dump(out, open(R/"metric_cis.json", "w"), indent=1); print("saved")
