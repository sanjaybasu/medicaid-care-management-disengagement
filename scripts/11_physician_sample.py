"""Step 11 (amendment v3.2 Section H): blinded physician review of clinical stakes, single phase.
Sample: from test-era eligible decision points, the last completed contact before disengagement for 100 patients who disengaged and the
last completed contact before program completion for 100 patients who completed the program (one index contact per patient; seed 20260909).
Each case is read by one of three physicians; 20% of cases are read by a second physician for agreement. Packets show the structured
summary as of the index contact, the notes of the prior 90 days, and the index note, with nothing after the index contact and no group label.
Items: open clinical need at the index contact (six columns) and harm if the patient had no further contact for 90 days (low, moderate, high).
Power: with 100 cases per group, a two-sided test at alpha 0.05 has 80% power to detect a difference of 15 percentage points in the share
with any open clinical need (0.75 versus 0.90). All packet files contain PHI and stay in data_cache/ (never versioned).
Writes data_cache/physician_review/{cases,forms}/..., ANSWER_KEY_do_not_share.csv, assignment.csv, results/physician_sample_meta.json (counts only)."""
import json, pathlib, re, html, numpy as np, pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
def clean(t):
    t = html.unescape(re.sub(r"<[^>]+>", " ", str(t or ""))); return re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", "\n", t)).strip()
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); PR = D/"physician_review"; SEED = 20260909; rng = np.random.default_rng(SEED); N_PER_GROUP = 100; DOUBLE = 0.20
for sub in ["cases", "forms"]: (PR/sub).mkdir(parents=True, exist_ok=True)
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet")
enc = pd.read_parquet(D/"encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date); enc = enc.dropna(subset=["enc_date"]).sort_values("enc_date")
df = F.merge(O[["decision_id","eligible","training_era","y_primary","grad_90","n_contacts_90"]].rename(columns={"training_era": "era"}), on="decision_id"); df["enc_date"] = pd.to_datetime(df.enc_date)
df = df[(df.eligible == 1) & (df.era == 0)].sort_values("enc_date")
dis = df[df.y_primary == 1].groupby("person_id").tail(1)                      # last contact before disengagement (no contact in the following 90 days)
com = df[(df.grad_90 == 1) & (df.y_primary == 0)].groupby("person_id").tail(1)  # last contact before program completion within 90 days
com = com[~com.person_id.isin(dis.person_id)]
pick = lambda d, lab: d.sample(n=N_PER_GROUP, random_state=int(rng.integers(1e9))).assign(group=lab)
S = pd.concat([pick(dis, "disengaged"), pick(com, "completed")]).reset_index(drop=True)
S["case_id"] = [f"R{k:04d}" for k in rng.permutation(np.arange(2000, 2000 + len(S)))]; S = S.sort_values("case_id").reset_index(drop=True)
S["reviewer"] = np.array(list("ABC"))[rng.permutation(np.arange(len(S)) % 3)]
dbl = S.groupby("group", group_keys=False).apply(lambda g: g.sample(frac=DOUBLE, random_state=SEED)).case_id
S["second_reviewer"] = np.where(S.case_id.isin(dbl), S.reviewer.map({"A": "B", "B": "C", "C": "A"}), "")
def band(a): return "under 18" if a < 18 else "18 to 44" if a < 45 else "45 to 64" if a < 65 else "65 and over"
COND = {"diabetes":"diabetes","htn":"hypertension","chf":"heart failure","copd":"COPD","sud":"substance use disorder","any_bh":"behavioral health condition","mdd":"depression","asthma":"asthma","polypharmacy":"polypharmacy","high_ed_ip":"high prior ED/inpatient use","no_pcp_last_10mo":"no primary care visit in prior 10 months"}
def packet(r):
    t = r.enc_date; g = enc[(enc.person_id == r.person_id) & (enc.enc_date < t) & (enc.enc_date >= t - pd.Timedelta(days=90))]
    conds = ", ".join(v for k, v in COND.items() if getattr(r, k) == 1) or "none flagged"; dsl = int(r.days_since_last_contact) if pd.notna(r.days_since_last_contact) else "none recorded"
    lines = [f"# Case {r.case_id}", "", f"**Structured summary (as of the index contact).** Age band {band(r.age)}; gender {r.gender}; state {str(r.state).title()}; race/ethnicity {r.race}; days since activation {int(r.days_from_zd)}; contacts in the prior 90 days {int(r.n_prior_90)}; days since last contact {dsl}; open goals {int(r.goals_open)}; condition flags: {conds}. Index contact by {r.cur_rg} ({r.contact_type}; {r.encounter_type}).", "", "## Notes of the prior 90 days (oldest first)", ""]
    if len(g) == 0: lines.append("_No contacts in the prior 90 days._")
    for e in g.itertuples(): lines += [f"**Day -{(t - e.enc_date).days}** ({e.roles}; {e.contact_type}; {e.encounter_type})", "", clean(e.note_text) or "_(empty note)_", ""]
    lines += ["## Index note", "", clean(r.note_text) or "_(empty note)_", ""]
    return "\n".join(lines)
for r in S.itertuples(): (PR/"cases"/f"case_{r.case_id}.md").write_text(packet(r))
FORM_COLS = ["q1_open_need_medication","q1_open_need_referral_or_appointment","q1_open_need_uncontrolled_condition","q1_open_need_social_crisis","q1_open_need_behavioral_health","q1_no_open_need","q2_harm_if_lost","comments"]
for rev in "ABC":
    sub = S[(S.reviewer == rev) | (S.second_reviewer == rev)]
    (PR/"cases"/f"reviewer_{rev}_all_cases.md").write_text("\n\n---\n\n".join(packet(r) for r in sub.itertuples()))
    pd.DataFrame({"case_id": sub.case_id, "reviewer": rev, **{c: "" for c in FORM_COLS}}).to_csv(PR/"forms"/f"reviewer_{rev}_form.csv", index=False)
S[["case_id","decision_id","person_id","group","reviewer","second_reviewer"]].to_csv(PR/"ANSWER_KEY_do_not_share.csv", index=False)
S[["case_id","reviewer","second_reviewer"]].to_csv(PR/"assignment.csv", index=False)
power = NormalIndPower().power(effect_size=proportion_effectsize(0.90, 0.75), nobs1=N_PER_GROUP, alpha=0.05, ratio=1.0)
meta = {"design": "single_phase_v2", "seed": SEED, "cases": int(len(S)), "per_group": N_PER_GROUP, "candidates": {"disengaged_patients": int(len(dis)), "completed_patients": int(len(com))},
        "forms_per_reviewer": {rev: int(((S.reviewer == rev) | (S.second_reviewer == rev)).sum()) for rev in "ABC"}, "double_read_cases": int((S.second_reviewer != "").sum()), "total_forms": int(len(S) + (S.second_reviewer != "").sum()),
        "power": {"alpha": 0.05, "two_sided": True, "p_disengaged": 0.90, "p_completed": 0.75, "difference": 0.15, "n_per_group": N_PER_GROUP, "power": round(float(power), 3)},
        "index_contact_rule": "last completed test-era contact before disengagement (no completed contact in the following 90 days) or before program completion within 90 days; one per patient"}
json.dump(meta, open(R/"physician_sample_meta.json", "w"), indent=1); print(json.dumps(meta, indent=1))
