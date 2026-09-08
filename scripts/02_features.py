"""Step 2: predictors measured strictly before or at t (PREREGISTRATION_v3 Section 5, excluding dense embeddings).
Output: data_cache/dp_features.parquet"""
import pandas as pd, numpy as np, re, pathlib, json
D = pathlib.Path("data_cache"); R = pathlib.Path("results")
sf = pd.read_parquet(D/"state_features.parquet"); sf["enc_date"] = pd.to_datetime(sf.enc_date); sf = sf.reset_index(drop=True)
enc = pd.read_parquet(D/"encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date); enc = enc.dropna(subset=["enc_date"])
goals = pd.read_parquet(D/"goals.parquet"); goals["upd"] = pd.to_datetime(goals.updated_at, errors="coerce"); goals["cre"] = pd.to_datetime(goals.created_at, errors="coerce")
m4 = pd.read_parquet("/Users/sanjaybasu/waymark-local/packaging/care-coordination-mechanism/cache/m4_classifications.parquet")
TAGS = ["referral_followthrough_complete","care_gap_resolved","med_adherence_barrier_addressed","social_barrier_resolved","urgent_issue_caught_acted"]
def rg(r):
    r = str(r); return "Therapist" if "Therap" in r else ("CPhT" if "Pharmacist Tech" in r else ("PharmD" if "Pharm" in r else ("CHW" if "CHW" in r else "CC")))
INP = {"HOME_VISIT","IN_COMMUNITY","PROVIDER_OFFICE","HOSPITAL","CBO","OTHER_INPERSON"}
enc["rg"] = enc.roles.map(rg); enc["inp"] = enc.contact_type.isin(INP).astype(int); enc["text_mod"] = enc.contact_type.isin(["SMS_TEXT","SMS_TEXT_CMT","EMAIL"]).astype(int)
enc = enc.merge(m4[["note_id"]+TAGS], left_on="encounter_id", right_on="note_id", how="left")
for t in TAGS: enc[t] = pd.to_numeric(enc[t], errors="coerce").fillna(0)
# engagement lexicon (pre-specified, Section 5c)
LEX = {"lex_unable_to_reach": r"unable to reach|could not reach|couldn't reach|no answer|not answer|unreachable|left (a )?(voice ?mail|message)|voicemail|vm left|no response|did not respond|didn't respond|no call ?back",
       "lex_bad_number": r"wrong number|disconnected|number (is )?(not|no longer) (in service|working)|invalid number|phone (is )?off",
       "lex_declined": r"declin|not interested|refus|does not want|doesn't want|do not want|no longer want|opt(ed)? out|stop (calling|texting)|unsubscribe|do not contact",
       "lex_fewer_contacts": r"fewer (calls|contacts)|less (often|frequent)|monthly check.?in|only (text|call)|prefers? (text|email|not)",
       "lex_moved": r"\bmoved\b|relocat|out of state|new address",
       "lex_incarcerated": r"incarcerat|jail|prison|detention",
       "lex_hospitalized": r"hospitali[sz]ed|admitted|in the hospital|inpatient|rehab facility|nursing facility|snf\b",
       "lex_busy_callback": r"busy|call back later|will call back|callback|reschedul|no.?show|missed appointment",
       "lex_positive_engagement": r"agreed|scheduled|will attend|confirmed|engaged|receptive|grateful|thank"}
txt = enc.note_text.fillna("").str.lower()
for k, pat in LEX.items(): enc[k] = txt.str.contains(pat, regex=True).astype(int)
E = {p: g.sort_values("enc_date") for p, g in enc.groupby("person_id")}
G = {p: g for p, g in goals.groupby("person_id")}
GOALCATS = {"SOCIAL": ["HOUSING_INSECURITY","HOUSING_QUALITY_SAFETY","FOOD_INSECURITY","FOOD_DIET_NUTRITION","TRANSPORTATION","UTILITIES","FINANCIAL","EMPLOYMENT","CHILDCARE","LEGAL","TECHNOLOGY","SOCIAL_CONNECTION","EDUCATION","VIOLENCE"],
            "CLINICAL": ["HYPERTENSION","DIABETES","ASTHMA_COPD","HEART_FAILURE","MEDICATION_ADHERENCE","MEDICATION_OPTIMIZATION"], "BH": ["MENTAL_HEALTH","OTHER_MENTAL_BEHAVIORAL","CARE_FOR_MH_BH","DEPRESSION","ANXIETY","SUBSTANCE_USE","ALCOHOL_USE"]}
D1 = np.timedelta64(1, "D"); rows = []
for r in sf.itertuples():
    t = r.enc_date.to_datetime64(); g = E.get(r.person_id); f = {"decision_id": r.encounter_id}
    if g is not None:
        d = g.enc_date.values; prior = g[d < t]; p90 = prior[prior.enc_date.values >= t - 90*D1]; p30 = prior[prior.enc_date.values >= t - 30*D1]
        cur = g[g.encounter_id == r.encounter_id]
        f.update(n_prior=len(prior), n_prior_90=len(p90), n_prior_30=len(p30), days_since_last_contact=float((t - prior.enc_date.values.max())/D1) if len(prior) else 999.0,
                 chw90=(p90.rg=="CHW").sum(), pharm90=(p90.rg=="PharmD").sum(), cpht90=(p90.rg=="CPhT").sum(), ther90=(p90.rg=="Therapist").sum(), cc90=(p90.rg=="CC").sum(),
                 inp90=p90.inp.sum(), textmod90=p90.text_mod.sum(), n_disc90=p90.rg.nunique(),
                 gap_mean=float(np.mean(np.diff(prior.enc_date.values[-6:]).astype("timedelta64[D]").astype(float))) if len(prior) > 1 else 999.0,
                 gap_max=float(np.max(np.diff(prior.enc_date.values[-6:]).astype("timedelta64[D]").astype(float))) if len(prior) > 1 else 999.0,
                 cur_inp=int(cur.inp.iloc[0]) if len(cur) else 0, cur_textmod=int(cur.text_mod.iloc[0]) if len(cur) else 0, cur_rg=cur.rg.iloc[0] if len(cur) else "u",
                 prev_rg=prior.rg.iloc[-1] if len(prior) else "none", note_len=float(len(str(cur.note_text.iloc[0]))) if len(cur) else 0.0)
        for k in LEX: f[k+"_cur"] = int(cur[k].iloc[0]) if len(cur) else 0; f[k+"_p90"] = int(p90[k].sum())
        for tg in TAGS: f["tag_"+tg+"_cur"] = int(cur[tg].iloc[0]) if len(cur) else 0; f["tag_"+tg+"_p90"] = int(p90[tg].sum())
    else:
        f.update(n_prior=0, n_prior_90=0, n_prior_30=0, days_since_last_contact=999.0, cur_rg="u", prev_rg="none")
    gg = G.get(r.person_id)
    if gg is not None:
        before = gg[(gg.cre.values <= t)]
        f["goals_open"] = int((before.status.isin(["ACTIVE","NOT_STARTED","SUGGESTED"])).sum()); f["goals_completed_before"] = int(((before.status=="COMPLETED") & (before.upd.values <= t)).sum())
        for k, cats in GOALCATS.items(): f[f"goals_{k.lower()}"] = int(before.category.isin(cats).sum())
    rows.append(f)
F = pd.DataFrame(rows).fillna(0)
keep = ["encounter_id","person_id","enc_date","training_era","state","market","age","gender","race","risk_percentile","tier1_flg","days_from_zd","days_since_last_adt"] + [f"adt_all_prior_{w}d" for w in [30,90,180,365]] + [c for c in ["diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip","no_pcp_last_10mo"] if c in sf.columns] + ["contact_type","encounter_type","note_text"]
out = sf[keep].rename(columns={"encounter_id":"decision_id"}).merge(F, on="decision_id")
out.to_parquet(D/"dp_features.parquet"); print("features:", out.shape)
lexrates = {k: round(float(out[k+"_cur"].mean()), 4) for k in LEX}; print("lexicon prevalence (current note):", lexrates)
json.dump({"n_features_rows": len(out), "lexicon_prevalence_current_note": lexrates, "lexicon_patterns": LEX}, open(R/"features_meta.json","w"), indent=1)
