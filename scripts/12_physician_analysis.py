"""Step 12 (amendment v3.2 Section H): analysis of the single-phase blinded physician review of clinical stakes.
Reads data_cache/physician_review/returned/reviewer_{A,B,C}_form.csv and ANSWER_KEY_do_not_share.csv. Writes results/physician_review.json
(aggregates only). For each item: share of cases with the need or with harm rated high, by group (disengaged versus completed the program),
with 95% percentile intervals from 1,000 case-bootstrap resamples, and the difference; Cohen's kappa on the double-read cases."""
import json, pathlib, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); PR = D/"physician_review"; RET = PR/"returned"; rng = np.random.default_rng(20260909)
key = pd.read_csv(PR/"ANSWER_KEY_do_not_share.csv")
forms = [pd.read_csv(RET/f"reviewer_{r}_form.csv") for r in "ABC" if (RET/f"reviewer_{r}_form.csv").exists()]
if not forms: raise SystemExit("no returned forms in data_cache/physician_review/returned/")
f = pd.concat(forms); f = f.dropna(subset=["q2_harm_if_lost"]); f = f[f.q2_harm_if_lost.astype(str).str.strip() != ""]
NEED = ["q1_open_need_medication","q1_open_need_referral_or_appointment","q1_open_need_uncontrolled_condition","q1_open_need_social_crisis","q1_open_need_behavioral_health","q1_no_open_need"]
for c in NEED: f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0).astype(int)
f["any_open_need"] = (f[NEED[:-1]].sum(axis=1) > 0).astype(int); f["harm_high"] = (f.q2_harm_if_lost.astype(str).str.strip().str.lower() == "high").astype(int)
ITEMS = NEED + ["any_open_need", "harm_high"]
primary = f.merge(key[["case_id","group","reviewer"]], on=["case_id","reviewer"])            # the assigned first reader's form
per_case = f.groupby("case_id")[ITEMS].mean().reset_index().merge(key[["case_id","group"]], on="case_id")  # mean over readers when double-read
def share_ci(g, col):
    v = g[col].values; bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
    return {"share": round(float(v.mean()), 4), "ci_95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], "n": int(len(v))}
out = {"design": "single_phase_v2", "forms": int(len(f)), "cases_with_form": int(per_case.case_id.nunique()), "groups": {}, "differences": {}, "agreement": {}, "per_reviewer": {}}
for grp, g in per_case.groupby("group"): out["groups"][grp] = {c: share_ci(g, c) for c in ITEMS}
a, b = per_case[per_case.group == "disengaged"], per_case[per_case.group == "completed"]
for c in ITEMS:
    bs = [a[c].values[rng.integers(0, len(a), len(a))].mean() - b[c].values[rng.integers(0, len(b), len(b))].mean() for _ in range(1000)]
    out["differences"][c] = {"risk_difference_disengaged_minus_completed": round(float(a[c].mean() - b[c].mean()), 4), "ci_95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
dbl = f.groupby("case_id").filter(lambda g: g.reviewer.nunique() == 2)
for c in ITEMS:
    w = dbl.pivot_table(index="case_id", columns="reviewer", values=c, aggfunc="first")
    pairs = [(x, y) for x, y in [("A","B"),("B","C"),("A","C")] if x in w and y in w]
    v1 = np.concatenate([w.loc[w[[x, y]].notna().all(axis=1), x].values for x, y in pairs]) if pairs else np.array([]); v2 = np.concatenate([w.loc[w[[x, y]].notna().all(axis=1), y].values for x, y in pairs]) if pairs else np.array([])
    out["agreement"][c] = {"n_double_read": int(len(v1)), "percent_agreement": round(float((v1 == v2).mean()), 3) if len(v1) else None, "kappa": round(float(cohen_kappa_score(v1, v2)), 3) if len(v1) and len(set(v1) | set(v2)) > 1 else None}
for rev, g in f.merge(key[["case_id","group"]], on="case_id").groupby("reviewer"): out["per_reviewer"][rev] = {"forms": int(len(g)), "any_open_need_by_group": g.groupby("group").any_open_need.mean().round(3).to_dict(), "harm_high_by_group": g.groupby("group").harm_high.mean().round(3).to_dict()}
json.dump(out, open(R/"physician_review.json", "w"), indent=1); print(json.dumps(out, indent=1))
