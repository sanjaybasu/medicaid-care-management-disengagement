"""Step 12 (amendment v3.1 Section B): analyze returned physician review forms. Run after all forms are in data_cache/physician_review/returned/.
Writes results/physician_review.json (aggregate only; no case-level data leaves data_cache)."""
import json, pathlib, itertools, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
D = pathlib.Path("data_cache/physician_review"); R = pathlib.Path("results"); RET = D/"returned"
key = pd.read_csv(D/"ANSWER_KEY_do_not_share.csv")
f1 = pd.concat([pd.read_csv(p) for p in RET.glob("reviewer_*_form.csv") if "_phase" not in p.name]); f2 = pd.concat([pd.read_csv(p) for p in RET.glob("reviewer_*_phase2_form.csv")]) if list(RET.glob("reviewer_*_phase2_form.csv")) else None
f3 = pd.concat([pd.read_csv(p) for p in RET.glob("*phase3_form*.csv")]) if list(RET.glob("*phase3_form*.csv")) else None
if f3 is not None: f3["pattern"] = f3.pattern.astype(str).str.strip()
def fleiss(table):  # table: cases x categories counts
    n = table.sum(1)[0]; P_i = ((table**2).sum(1) - n) / (n*(n-1)); Pbar = P_i.mean(); pj = table.sum(0)/table.sum(); Pe = (pj**2).sum(); return float((Pbar - Pe)/(1 - Pe)) if Pe < 1 else float("nan")
def pairwise_kappa(df, item, minn=10):
    ks = []
    for a, b in itertools.combinations("ABC", 2):
        m = df[df.reviewer == a][["case_id", item]].merge(df[df.reviewer == b][["case_id", item]], on="case_id", suffixes=("_a","_b")).dropna()
        if len(m) < minn: continue
        cats = sorted(set(m[item+"_a"]) | set(m[item+"_b"])); po = (m[item+"_a"] == m[item+"_b"]).mean(); pe = sum((m[item+"_a"] == c).mean()*(m[item+"_b"] == c).mean() for c in cats); ks.append({"pair": a+b, "n": int(len(m)), "kappa": round(float((po-pe)/(1-pe)), 3) if pe < 1 else None})
    return ks
def icc(df, item):  # one-way random ICC(1) on cases with >=2 raters
    g = df.dropna(subset=[item]).groupby("case_id")[item]; g = g.filter(lambda s: len(s) >= 2); d = df.loc[g.index].dropna(subset=[item]); grp = d.groupby("case_id")[item]
    k = grp.size().mean(); msb = grp.mean().var(ddof=1) * k; msw = grp.var(ddof=1).mean(); return round(float((msb - msw)/(msb + (k-1)*msw)), 3)
out = {"phase1_forms": int(len(f1)), "cases_reviewed": int(f1.case_id.nunique())}
out["agreement"] = {it: pairwise_kappa(f1, it) for it in ["q1_engagement_status","q4_harm_if_lost","q5_stated_intent"]}
out["icc_prognosis"] = icc(f1, "q2_prognosis_1to5")
core = f1.groupby("case_id").reviewer.nunique(); core = core[core == 3].index
if len(core) >= 10:
    sub = f1[f1.case_id.isin(core)]; cats = sorted(sub.q1_engagement_status.dropna().unique()); tab = np.array([[ (sub[sub.case_id == c].q1_engagement_status == k).sum() for k in cats] for c in core]); out["fleiss_kappa_q1_core"] = round(fleiss(tab), 3)
# physician prognosis vs model on the same cases (mean prognosis per case; reversed so higher = more likely to disengage)
m = f1.groupby("case_id").q2_prognosis_1to5.mean().reset_index().merge(key, on="case_id"); m["phys_risk"] = 6 - m.q2_prognosis_1to5
rng = np.random.default_rng(20260908); diffs = []
for _ in range(1000):
    ix = rng.integers(0, len(m), len(m)); b = m.iloc[ix]
    if b.y.nunique() == 2: diffs.append(roc_auc_score(b.y, b.M5_full) - roc_auc_score(b.y, b.phys_risk))
out["prognosis_auroc_within_stratum_all"] = {st: round(float(roc_auc_score(gg.y, gg.phys_risk)), 3) for st, gg in m.groupby("stratum") if gg.y.nunique() == 2}; out["model_auroc_within_stratum"] = {st: round(float(roc_auc_score(gg.y, gg.M5_full)), 3) for st, gg in m.groupby("stratum") if gg.y.nunique() == 2}
out["prognosis_vs_model"] = {"n_cases": int(len(m)), "auroc_physician": round(float(roc_auc_score(m.y, m.phys_risk)), 3), "auroc_model": round(float(roc_auc_score(m.y, m.M5_full)), 3), "difference_model_minus_physician": round(float(roc_auc_score(m.y, m.M5_full) - roc_auc_score(m.y, m.phys_risk)), 3), "ci_95": [round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)], "note": "sampled cases are stratified by risk and outcome; AUROC is comparable between raters on the same cases but not to the population AUROC"}
need_cols = [c for c in f1.columns if c.startswith("q3_")]; agg = f1.groupby("case_id")[need_cols + ["q2_prognosis_1to5"]].mean().reset_index().merge(key[["case_id","y","stratum"]], on="case_id")
agg["harm_high"] = f1.assign(h=(f1.q4_harm_if_lost == "high").astype(float)).groupby("case_id").h.mean().values
out["open_need_and_harm_by_outcome"] = {f"y{yv}": {c: round(float(agg.loc[agg.y == yv, c].mean()), 3) for c in need_cols + ["harm_high"]} for yv in (1, 0)}
out["engagement_status_by_outcome"] = {f"y{yv}": f1[f1.case_id.isin(key.loc[key.y == yv, "case_id"])].q1_engagement_status.value_counts(normalize=True).round(3).to_dict() for yv in (1, 0)}
if f2 is not None:
    out["phase2_reasons"] = f2.drop_duplicates("case_id").reason.value_counts(normalize=True).round(3).to_dict(); out["phase2_n"] = int(f2.case_id.nunique()); out["phase2_double_coded_kappa"] = pairwise_kappa(f2, "reason", minn=3); dd = f2.groupby("case_id").filter(lambda g: len(g) == 2); out["phase2_double_coded_agreement"] = {"n_double_coded": int(dd.case_id.nunique()), "percent_agreement": round(float(dd.groupby("case_id").reason.nunique().eq(1).mean()), 3) if len(dd) else None}
if f3 is not None:
    out["phase3_pattern_ratings"] = f3.groupby("pattern").clinical_meaningfulness_1to5.agg(["mean","std","count"]).round(2).reset_index().to_dict("records")

# per-reviewer response profiles and reviewer-level prognosis AUROC (data-quality check: a reviewer whose answers are constant or rule-like is flagged)
out["per_reviewer"] = {}
for rev, g in f1.groupby("reviewer"):
    km = g.merge(key, on="case_id"); rec = {"n": int(len(g)), "q1_distribution": g.q1_engagement_status.value_counts(normalize=True).round(3).to_dict(), "q2_distribution": g.q2_prognosis_1to5.value_counts(normalize=True).round(3).to_dict(), "q4_distribution": g.q4_harm_if_lost.value_counts(normalize=True).round(3).to_dict(), "q5_distribution": g.q5_stated_intent.value_counts(normalize=True).round(3).to_dict(),
           "share_no_open_need": round(float(g.q3_no_open_need.mean()), 3), "share_uncontrolled_condition": round(float(g.q3_open_need_uncontrolled_condition.mean()), 3),
           "prognosis_auroc_vs_outcome": round(float(roc_auc_score(km.y, 6 - km.q2_prognosis_1to5)), 3) if km.y.nunique() == 2 else None, "model_auroc_same_cases": round(float(roc_auc_score(km.y, km.M5_full)), 3) if km.y.nunique() == 2 else None,
           "q1_q5_cross_tab": {f"{a}|{b}": int(v) for (a, b), v in g.groupby(["q1_engagement_status","q5_stated_intent"]).size().items()}}
    n_items = len([c for c in g.columns if c.startswith("q")]); const = [c for c in g.columns if c.startswith("q") and g[c].nunique() == 1]; rec["constant_items"] = const; out["per_reviewer"][rev] = rec
# sensitivity: agreement and prognosis excluding each reviewer in turn
out["leave_one_reviewer_out"] = {}
for rev in "ABC":
    sub = f1[f1.reviewer != rev]; m2 = sub.groupby("case_id").q2_prognosis_1to5.mean().reset_index().merge(key, on="case_id"); m2["phys_risk"] = 6 - m2.q2_prognosis_1to5
    out["leave_one_reviewer_out"][f"without_{rev}"] = {"n_cases": int(len(m2)), "auroc_physician": round(float(roc_auc_score(m2.y, m2.phys_risk)), 3), "auroc_model": round(float(roc_auc_score(m2.y, m2.M5_full)), 3), "kappa_q1": pairwise_kappa(sub, "q1_engagement_status"), "kappa_q4": pairwise_kappa(sub, "q4_harm_if_lost")}
# reasons by reviewer (phase 2)
if f2 is not None: out["phase2_reasons_by_reviewer"] = {rev: g.reason.value_counts(normalize=True).round(3).to_dict() for rev, g in f2.groupby("reviewer")}
json.dump(out, open(R/"physician_review.json", "w"), indent=1, default=str)

# data-quality rule (applied after inspection, documented in the deviation log): a phase 1 form with any constant item across all cases, or with a
# single repeated free-text comment across all cases, is not treated as an independent review; results are reported with all forms and with that form excluded
flag = {rev: (len(out["per_reviewer"][rev]["constant_items"]) > 0) or (f1[f1.reviewer == rev].comments.nunique() <= 1) for rev in out["per_reviewer"]}
out["reviewer_quality_flags"] = flag; excl = [r for r, v in flag.items() if v]
if excl:
    g = f1[~f1.reviewer.isin(excl)]; agg2 = g.groupby("case_id")[need_cols + ["q2_prognosis_1to5"]].mean().reset_index().merge(key[["case_id","y","stratum"]], on="case_id")
    agg2["harm_high"] = g.assign(h=(g.q4_harm_if_lost == "high").astype(float)).groupby("case_id").h.mean().values
    out["excluding_flagged"] = {"excluded": excl, "n_forms": int(len(g)), "cases_with_at_least_one_review": int(g.case_id.nunique()),
        "open_need_and_harm_by_outcome": {f"y{yv}": {c: round(float(agg2.loc[agg2.y == yv, c].mean()), 3) for c in need_cols + ["harm_high"]} for yv in (1, 0)},
        "any_open_need_by_outcome": {f"y{yv}": round(float((agg2.loc[agg2.y == yv, "q3_no_open_need"] < 0.5).mean()), 3) for yv in (1, 0)},
        "engagement_status_by_outcome": {f"y{yv}": g[g.case_id.isin(key.loc[key.y == yv, "case_id"])].q1_engagement_status.value_counts(normalize=True).round(3).to_dict() for yv in (1, 0)},
        "icc_prognosis": icc(g, "q2_prognosis_1to5")}
    m3 = g.groupby("case_id").q2_prognosis_1to5.mean().reset_index().merge(key, on="case_id"); m3["phys_risk"] = 6 - m3.q2_prognosis_1to5
    d3 = []
    for _ in range(1000):
        ix = rng.integers(0, len(m3), len(m3)); b = m3.iloc[ix]
        if b.y.nunique() == 2: d3.append(roc_auc_score(b.y, b.M5_full) - roc_auc_score(b.y, b.phys_risk))
    out["excluding_flagged"]["prognosis_vs_model"] = {"n_cases": int(len(m3)), "auroc_physician": round(float(roc_auc_score(m3.y, m3.phys_risk)), 3), "auroc_model": round(float(roc_auc_score(m3.y, m3.M5_full)), 3), "difference_model_minus_physician_ci_95": [round(float(np.percentile(d3, 2.5)), 3), round(float(np.percentile(d3, 97.5)), 3)]}
    # within-stratum prognosis: does physician prognosis separate outcome within each model risk stratum?
    out["excluding_flagged"]["prognosis_auroc_within_stratum"] = {st: round(float(roc_auc_score(gg.y, gg.phys_risk)), 3) for st, gg in m3.groupby("stratum") if gg.y.nunique() == 2}
    if f2 is not None:
        g2 = f2[~f2.reviewer.isin(excl)].drop_duplicates("case_id"); out["excluding_flagged"]["phase2_reasons"] = g2.reason.value_counts(normalize=True).round(3).to_dict(); out["excluding_flagged"]["phase2_n"] = int(len(g2))
    if f3 is not None:
        g3 = f3.copy(); g3["reviewer"] = np.repeat(["A","B","C"], [len(pd.read_csv(p)) for p in sorted(RET.glob("*phase3_form*.csv"))])[:len(g3)] if "reviewer" not in g3.columns else g3["reviewer"]
out["design_note"] = "Cases were sampled with near-equal numbers of disengaged and retained members within each model risk stratum, so AUROC on this sample measures discrimination within strata only and is not comparable to the population AUROC of 0.897; the model's own AUROC on these cases is reported for that reason."
out["any_open_need_by_outcome_all_reviewers"] = {f"y{yv}": round(float((agg.loc[agg.y == yv, "q3_no_open_need"] < 0.5).mean()), 3) for yv in (1, 0)}
json.dump(out, open(R/"physician_review.json", "w"), indent=1, default=str)
print(json.dumps({k: out.get(k, "None") for k in ["reviewer_quality_flags","excluding_flagged","any_open_need_by_outcome_all_reviewers","leave_one_reviewer_out","phase2_reasons_by_reviewer"]}, indent=1, default=str))
