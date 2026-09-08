"""Step 8: assemble results/canonical.json and render every figure and table from it (PREREGISTRATION_v3 Section 9).
No literal result appears in this file; every number is read from results/*.json or computed from data_cache/*.parquet at render time."""
import json, pathlib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); NB = pathlib.Path("../../notebooks/care-management-disengagement"); FIG = NB/"figures"; TAB = NB/"tables"
J = lambda f: json.load(open(R/f)) if (R/f).exists() else None
flow, models, subs, lev, hp, fmeta, seq, sens = J("flow.json"), J("models_primary.json"), J("subgroups.json"), J("levers.json"), J("hyperparameters.json"), J("features_meta.json"), J("sequence_models.json"), J("sensitivity.json")
FULL = "M5_full" if "M5_full" in models["models"] else "M3b_plus_lexicon_tags"
NICE = {"VIRGINIA":"Virginia","WASHINGTON":"Washington","OHIO":"Ohio","CHW":"Community health worker","CC":"Care coordinator","CPhT":"Pharmacy technician","PharmD":"Clinical pharmacist","Therapist":"Therapist"}
FEAT = {"chw90":"CHW contacts, prior 90 d","note_len":"Note length","cur_rg_CHW":"Contact by CHW","lex_declined_cur":"Lexicon: declined (current note)","encounter_type_MEET_THE_PATIENT":"Encounter type: meet the patient","n_prior_90":"Contacts, prior 90 d","goals_open":"Open goals","lex_positive_engagement_p90":"Lexicon: positive engagement (prior 90 d)","lex_positive_engagement_cur":"Lexicon: positive engagement (current note)","goals_clinical":"Clinical goals","goals_social":"Social goals","risk_percentile":"Acute-care risk percentile","n_prior_30":"Contacts, prior 30 d","encounter_type_PROVIDER_COORDINATION":"Encounter type: provider coordination","days_since_last_adt":"Days since last acute event","n_prior":"Contacts, cumulative","days_since_last_contact":"Days since last contact","cur_rg_CC":"Contact by care coordinator","cur_rg_CPhT":"Contact by pharmacy technician","encounter_type_PATIENT_OUTREACH":"Encounter type: patient outreach","days_from_zd":"Days since activation","gap_mean":"Mean inter-contact gap","gap_max":"Longest inter-contact gap","age":"Age","cc90":"Care coordinator contacts, prior 90 d","cpht90":"Pharmacy technician contacts, prior 90 d","n_disc90":"Distinct disciplines, prior 90 d","textmod90":"Text/email contacts, prior 90 d","inp90":"In-person contacts, prior 90 d","goals_completed_before":"Goals completed before t","cur_textmod":"Current contact by text/email","cur_inp":"Current contact in person","pharm90":"Pharmacist contacts, prior 90 d","ther90":"Therapist contacts, prior 90 d","tier1_flg":"Tier 1 flag","goals_bh":"Behavioral health goals"}
LABEL = {"M0_signal_risk": "M0 acute-care risk score", "M1_structured": "M1 structured", "M2_structured_history": "M2 structured + contact history", "M3_plus_tfidf_text": "M3 + note text (TF-IDF)", "M3b_plus_lexicon_tags": "M3b + lexicon and tags", "M4_plus_embeddings": "M4 + note embeddings", "M5_full": "M5 full (text, embeddings, lexicon, tags)", "text_only_tfidf": "Note text alone (TF-IDF)", "embeddings_only": "Note embeddings alone"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"M0_signal_risk": "#9e9e9e", "M1_structured": "#7fb3d5", "M2_structured_history": "#2e6f9e", "M3_plus_tfidf_text": "#e08e45", FULL: "#b2182b"}
# ---------------- data
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet"); df = F.merge(O.drop(columns=["person_id","enc_date","training_era","state"]), on="decision_id")
el = df[df.eligible == 1].copy(); P = pd.read_parquet(D/"test_predictions.parquet")
INP = ["HOME_VISIT","IN_COMMUNITY","PROVIDER_OFFICE","HOSPITAL","CBO","OTHER_INPERSON"]; TXT = ["SMS_TEXT","SMS_TEXT_CMT","EMAIL"]; VID = ["VIDEO_VISIT"]
def modality(ct): return "In person" if ct in INP else ("Text or email" if ct in TXT else ("Video" if ct in VID else "Telephone or other remote"))
el["modality"] = el.contact_type.map(modality)
# ---------------- Table 1 (computed)
def pct(s): return f"{100*s.mean():.1f}"
def t1_block(d):
    first = d.sort_values("enc_date").groupby("person_id").first()
    rows = {"Decision points (completed contacts), n": f"{len(d):,}", "Members, n": f"{d.person_id.nunique():,}", "Contacts per member, median (IQR)": f"{d.groupby('person_id').size().median():.0f} ({d.groupby('person_id').size().quantile(.25):.0f}-{d.groupby('person_id').size().quantile(.75):.0f})",
            "Age at first contact, mean (SD), years": f"{first.age.mean():.1f} ({first.age.std():.1f})", "Female, % of members": pct(first.gender == "Female"),
            "Black or African American, % of members": pct(first.race == "Black or African American"), "White, % of members": pct(first.race == "White"), "Hispanic, % of members": pct(first.race == "Hispanic"), "Asian, % of members": pct(first.race == "Asian"), "Other or unknown race, % of members": pct(~first.race.isin(["Black or African American","White","Hispanic","Asian"])),
            "Virginia, % of members": pct(first.state == "VIRGINIA"), "Washington, % of members": pct(first.state == "WASHINGTON"), "Ohio, % of members": pct(first.state == "OHIO"),
            "Risk percentile at activation, median (IQR)": f"{first.risk_percentile.median():.0f} ({first.risk_percentile.quantile(.25):.0f}-{first.risk_percentile.quantile(.75):.0f})",
            "Any behavioral health condition, % of members": pct(first.any_bh == 1), "Substance use disorder, % of members": pct(first.sud == 1), "Hypertension, % of members": pct(first.htn == 1), "Diabetes, % of members": pct(first.diabetes == 1), "Asthma or COPD, % of members": pct((first.asthma == 1) | (first.copd == 1)), "Polypharmacy, % of members": pct(first.polypharmacy == 1), "No primary care visit in prior 10 months, % of members": pct(first.no_pcp_last_10mo == 1),
            "Days since activation at contact, median (IQR)": f"{d.days_from_zd.median():.0f} ({d.days_from_zd.quantile(.25):.0f}-{d.days_from_zd.quantile(.75):.0f})",
            "Contact by community health worker, % of contacts": pct(d.cur_rg == "CHW"), "Contact by care coordinator, % of contacts": pct(d.cur_rg == "CC"), "Contact by pharmacy technician, % of contacts": pct(d.cur_rg == "CPhT"), "Contact by clinical pharmacist, % of contacts": pct(d.cur_rg == "PharmD"), "Contact by therapist, % of contacts": pct(d.cur_rg == "Therapist"),
            "Telephone or other remote contact, % of contacts": pct(d.modality == "Telephone or other remote"), "Text or email contact, % of contacts": pct(d.modality == "Text or email"), "Video contact, % of contacts": pct(d.modality == "Video"), "In-person contact, % of contacts": pct(d.modality == "In person"),
            "Note length, median (IQR), characters": f"{d.note_len.median():.0f} ({d.note_len.quantile(.25):.0f}-{d.note_len.quantile(.75):.0f})", "Prior contacts in 90 days, median (IQR)": f"{d.n_prior_90.median():.0f} ({d.n_prior_90.quantile(.25):.0f}-{d.n_prior_90.quantile(.75):.0f})",
            "Disengagement within 90 days, % of contacts": pct(d.y_primary == 1), "Explicit exit status within 90 days, % of contacts": pct(d.y_explicit == 1), "Silent loss within 90 days, % of contacts": pct(d.y_silent == 1), "No contact within 30 days, % of contacts": pct(d.y_30 == 1), "Graduation within 90 days, % of contacts": pct(d.grad_90 == 1)}
    return rows
t1 = {"Training era (activated through 2025-06-30)": t1_block(el[el.training_era == 1]), "Test era (activated 2025-07-01 or later)": t1_block(el[el.training_era == 0]), "All": t1_block(el)}
t1df = pd.DataFrame(t1); t1df.index.name = "Characteristic"; t1df.to_csv(TAB/"table1_cohort.csv")
with open(TAB/"table1_cohort.md", "w") as f:
    f.write("**Table 1. Members and decision points by era.** Member characteristics are taken at each member's first eligible contact; contact characteristics are over all eligible decision points (completed care-team contacts within 365 days of activation with at least 60 enrolled days in the following 90 days).\n\n| Characteristic | " + " | ".join(t1df.columns) + " |\n|---|" + "---|"*len(t1df.columns) + "\n")
    for i, r in t1df.iterrows(): f.write(f"| {i} | " + " | ".join(r.values) + " |\n")
# ---------------- derived numbers for canonical
M = models["models"]; C = models["contrasts"]
def per100(m, k): return round(100*M[m][f"ppv_flag{k}"], 1)
canon = {"generated_from": ["flow.json","models_primary.json","subgroups.json","levers.json","hyperparameters.json","features_meta.json","sequence_models.json","sensitivity.json"], "full_model": FULL, "flow": flow, "models": models, "subgroups": subs, "levers": lev, "hyperparameters": hp, "features_meta": fmeta, "sequence_models": seq, "sensitivity": sens,
         "table1": t1, "derived": {"disengagements_per_100_flags": {m: {f"flag{k}": per100(m, k) for k in (10,20,30)} for m in M},
         "auprc_relative_gain_full_vs_M2_pct": round(100*C["full_vs_structured_history"]["auprc"]["diff"]/M["M2_structured_history"]["auprc"], 1),
         "auprc_relative_gain_M2_vs_M0_pct": round(100*C["structured_history_vs_signal"]["auprc"]["diff"]/M["M0_signal_risk"]["auprc"], 1),
         "auprc_over_prevalence_full": round(M[FULL]["auprc"]/models["test_event_rate"], 2), "auprc_over_prevalence_M0": round(M["M0_signal_risk"]["auprc"]/models["test_event_rate"], 2),
         "n_flagged_at_20pct": int(round(0.20*models["n_test"])), "events_captured_at_20pct_full": int(round(M[FULL]["sens_flag20"]*models["n_test"]*models["test_event_rate"])), "events_captured_at_20pct_M2": int(round(M["M2_structured_history"]["sens_flag20"]*models["n_test"]*models["test_event_rate"])), "events_captured_at_20pct_M0": int(round(M["M0_signal_risk"]["sens_flag20"]*models["n_test"]*models["test_event_rate"])),
         "test_events": int(round(models["n_test"]*models["test_event_rate"])), "primary_events_all": int(el.y_primary.sum()), "explicit_events_all": int(el.y_explicit.sum()),
         "subgroup_auroc_gain_all_positive_state": all(v["auroc_gain"] > 0 for v in subs["by_state"].values()), "subgroup_auroc_gain_all_positive_race": all(v["auroc_gain"] > 0 for v in subs["by_race"].values()),
         "subgroup_auroc_gain_range_state": [min(v["auroc_gain"] for v in subs["by_state"].values()), max(v["auroc_gain"] for v in subs["by_state"].values())], "subgroup_auroc_gain_range_race": [min(v["auroc_gain"] for v in subs["by_race"].values()), max(v["auroc_gain"] for v in subs["by_race"].values())],
         "loso_full_auroc_range": [min(v["full_text"]["auroc"] for v in subs["leave_one_state_out"].values()), max(v["full_text"]["auroc"] for v in subs["leave_one_state_out"].values())], "loso_auprc_gain": {k: round(v["full_text"]["auprc"] - v["structured_history"]["auprc"], 3) for k, v in subs["leave_one_state_out"].items()},
         "explicit_share_of_primary_pct": round(100*flow["primary_event_composition"]["explicit_status"], 1), "administrative_share_of_primary_pct": round(100*flow["primary_event_composition"]["administrative_status"], 1),
         "neither_status_share_pct": round(100*(1 - flow["primary_event_composition"]["explicit_status"] - flow["primary_event_composition"]["administrative_status"] + float(((el.y_primary==1)&(el.dis_status_90==1)&(el.admin_90==1)).sum()/max((el.y_primary==1).sum(),1))), 1),
         "primary_rate_by_days_since_activation": {lab: round(float(el.loc[(el.days_from_zd > lo) & (el.days_from_zd <= hi), "y_primary"].mean()), 4) for lab, lo, hi in [("0_30",-1,30),("31_90",30,90),("91_180",90,180),("181_365",180,366)]},
         "primary_rate_by_discipline": el.groupby("cur_rg").y_primary.mean().round(4).to_dict(), "primary_rate_by_encounter_type": {k: {"rate": round(float(v["mean"]), 4), "n": int(v["size"])} for k, v in el.groupby("encounter_type").y_primary.agg(["mean","size"]).iterrows() if v["size"] >= 500}, "share_first30_by_discipline": {rg: round(float((g.days_from_zd <= 30).mean()), 4) for rg, g in el.groupby("cur_rg")}, "outreach_notes_unreached_share": round(float(el.loc[el.encounter_type == "PATIENT_OUTREACH", "note_text"].fillna("").str.lower().str.contains(r"voicemail|left (a )?message|\blvm\b|no answer|did not answer|unable to reach|unsuccessful").mean()), 4), "events_captured_gain_full_vs_M2_at_20pct": int(round(M[FULL]["sens_flag20"]*models["n_test"]*models["test_event_rate"])) - int(round(M["M2_structured_history"]["sens_flag20"]*models["n_test"]*models["test_event_rate"])), "lever_L1_mde_or_80pct_power": round(float(np.exp(2.802 * (np.log(lev["exposures"]["L1_inperson"]["within_patient_fe"]["ci_95"][1]) - np.log(lev["exposures"]["L1_inperson"]["within_patient_fe"]["ci_95"][0])) / 3.92)), 2), "min_auroc_full_after_day30": min(v["full"]["auroc"] for k, v in subs["by_days_since_activation"].items() if k != "0_30"), "race_share_other_unknown_pct": round(100*float((~el.sort_values("enc_date").groupby("person_id").first().race.isin(["Black or African American","White","Hispanic","Asian"])).mean()), 1), "primary_rate_by_discipline_first30_vs_later": {f"{rg}_{'first30' if e else 'later'}": round(float(r), 4) for (e, rg), r in el.groupby([el.days_from_zd <= 30, "cur_rg"]).y_primary.mean().items()}, "share_of_contacts_first30": round(float((el.days_from_zd <= 30).mean()), 4), "share_of_events_first30": round(float((el.loc[el.y_primary == 1, "days_from_zd"] <= 30).mean()), 4), "event_composition_direct": {"explicit": round(float((el.loc[el.y_primary==1,"dis_status_90"]==1).mean()),3), "administrative": round(float((el.loc[el.y_primary==1,"admin_90"]==1).mean()),3), "both": round(float(((el.loc[el.y_primary==1,"dis_status_90"]==1)&(el.loc[el.y_primary==1,"admin_90"]==1)).mean()),3), "neither": round(float(((el.loc[el.y_primary==1,"dis_status_90"]==0)&(el.loc[el.y_primary==1,"admin_90"]==0)).mean()),3)}, "primary_rate_by_modality": el.groupby("modality").y_primary.mean().round(4).to_dict(),
         "contacts_total": int(len(F)), "notes_unique_embedded": None, "study_window": [str(F.enc_date.min().date()), str(F.enc_date.max().date())], "test_contact_window": [str(el[el.training_era==0].enc_date.min().date()), str(el[el.training_era==0].enc_date.max().date())],
         "patients_total": int(F.person_id.nunique()), "eligible_patients": int(el.person_id.nunique()), "train_patients": int(el[el.training_era==1].person_id.nunique()), "test_patients": int(el[el.training_era==0].person_id.nunique()),
         "excluded_dps": int(len(F) - flow["eligible_60_enrolled_days"]), "excluded_pct": round(100*(len(F) - flow["eligible_60_enrolled_days"])/len(F), 1)}}
if (R/"embed_log.txt").exists():
    for line in open(R/"embed_log.txt"):
        if "unique" in line: canon["derived"]["notes_unique_embedded"] = int(line.split("unique")[0].split(",")[-1].strip().split()[0])
if seq: canon["derived"]["sequence_best"] = max(seq["models"].items(), key=lambda kv: kv[1]["auprc"])[0]; canon["derived"]["sequence_auprc_range"] = [min(v["auprc"] for v in seq["models"].values()), max(v["auprc"] for v in seq["models"].values())]; canon["derived"]["sequence_auroc_range"] = [min(v["auroc"] for v in seq["models"].values()), max(v["auroc"] for v in seq["models"].values())]
json.dump(canon, open(R/"canonical.json", "w"), indent=1, default=str)
# ---------------- Figure 1: flow (all counts from flow.json / canonical)
fig, ax = plt.subplots(figsize=(9.2, 3.9)); ax.axis("off"); ax.set_xlim(0,1.1); ax.set_ylim(0.15,1.0); d = canon["derived"]
boxes = [(0.42, 0.90, f"Completed care-team contacts within 365 days of activation\n{flow['decision_points']:,} decision points; {flow['patients']:,} members; {d['study_window'][0]} to {d['study_window'][1]}"),
         (0.42, 0.64, f"Eligible: at least 60 enrolled days in the following 90 days\n{flow['eligible_60_enrolled_days']:,} decision points ({flow['eligible_pct']}%); {d['eligible_patients']:,} members"),
         (0.20, 0.34, f"Training era\n(activated through 2025-06-30)\n{flow['eligible_train']:,} decision points\n{d['train_patients']:,} members"),
         (0.64, 0.34, f"Test era\n(activated 2025-07-01 or later)\n{flow['eligible_test']:,} decision points\n{flow['eligible_test_patients']:,} members\n{flow['primary_events_test']:,} disengagements ({100*flow['primary_rate_test']:.1f}%)"),
         (0.93, 0.64, f"Excluded: fewer than\n60 enrolled days in the\nfollowing 90 days\n({d['excluded_dps']:,}; {d['excluded_pct']}%)")]
for x, y, t in boxes: ax.text(x, y, t, ha="center", va="center", fontsize=8.3, bbox=dict(boxstyle="round,pad=0.5", fc="#f4f6f8", ec="#555"))
for (x0,y0),(x1,y1) in [((0.42,0.84),(0.42,0.71)),((0.42,0.57),(0.20,0.45)),((0.42,0.57),(0.64,0.45)),((0.42,0.77),(0.80,0.66))]: ax.annotate("", xy=(x1,y1), xytext=(x0,y0), arrowprops=dict(arrowstyle="->", color="#555"))
fig.savefig(FIG/"fig1_flow.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ---------------- Figure 2: discrimination and calibration
y = P.y.values; order = ["M0_signal_risk","M1_structured","M2_structured_history","M3_plus_tfidf_text",FULL]
fig, axs = plt.subplots(1, 3, figsize=(11, 4.6))
for m in order:
    fpr, tpr, _ = roc_curve(y, P[m]); pr, rc, _ = precision_recall_curve(y, P[m])
    axs[0].plot(fpr, tpr, color=COL[m], lw=1.4, label=f"{LABEL[m]} ({M[m]['auroc']:.3f})"); axs[1].plot(rc, pr, color=COL[m], lw=1.4, label=f"{LABEL[m]} ({M[m]['auprc']:.3f})")
axs[0].plot([0,1],[0,1], ls=":", c="k", lw=0.8); axs[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title="a  Receiver operating characteristic"); axs[0].legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
axs[1].axhline(y.mean(), ls=":", c="k", lw=0.8); axs[1].set(xlabel="Sensitivity (recall)", ylabel="Positive predictive value", title="b  Precision-recall"); axs[1].legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
for m in ["M2_structured_history", FULL]:
    q = pd.qcut(P[m], 10, labels=False, duplicates="drop"); g = pd.DataFrame({"p": P[m], "y": y, "q": q}).groupby("q").agg(p=("p","mean"), y=("y","mean"), n=("y","size"))
    se = np.sqrt(g.y*(1-g.y)/g.n); axs[2].errorbar(g.p, g.y, yerr=1.96*se, fmt="o-", ms=3.5, lw=1.2, color=COL[m], label=f"{LABEL[m]} (slope {M[m]['calibration_slope']:.2f})")
axs[2].plot([0,1],[0,1], ls=":", c="k", lw=0.8); axs[2].set(xlabel="Predicted probability (decile mean)", ylabel="Observed disengagement", title="c  Calibration, test era", xlim=(0,1), ylim=(0,1)); axs[2].legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
fig.tight_layout(); fig.savefig(FIG/"fig2_discrimination_calibration.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ---------------- Figure 3: decision curves and disengagements per 100 flags
def nb_at(yv, pv, pt): flag = pv >= pt; n = len(yv); tp = (flag & (yv==1)).sum(); fp = (flag & (yv==0)).sum(); return tp/n - fp/n*(pt/(1-pt))
th = np.linspace(0.05, 0.60, 56); fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for m in ["M0_signal_risk","M2_structured_history",FULL]: axs[0].plot(th, [nb_at(y, P[m].values, t) for t in th], color=COL[m], lw=1.4, label=LABEL[m])
axs[0].plot(th, [y.mean() - (1-y.mean())*t/(1-t) for t in th], c="k", ls="--", lw=0.9, label="Flag every contact"); axs[0].axhline(0, c="k", ls=":", lw=0.8, label="Flag none"); axs[0].set(xlabel="Threshold probability of disengagement", ylabel="Net benefit (per contact)", title="a  Decision curves, test era", ylim=(-0.02, 0.2)); axs[0].legend(fontsize=6.5, frameon=False)
fr = np.arange(0.05, 0.51, 0.05)
for m in ["M0_signal_risk","M2_structured_history",FULL]:
    axs[1].plot(100*fr, [100*(P[m] >= np.quantile(P[m], 1-f)).values.astype(bool).__and__(y==1).sum()/max((P[m] >= np.quantile(P[m], 1-f)).sum(),1) for f in fr], "o-", ms=3.5, color=COL[m], lw=1.4, label=LABEL[m])
axs[1].axhline(100*y.mean(), c="k", ls=":", lw=0.8, label="Base rate"); axs[1].set(xlabel="Contacts flagged, %", ylabel="Disengagements per 100 flagged contacts", title="b  Yield of flags"); axs[1].legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig(FIG/"fig3_decision_curves.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ---------------- Figure 4: subgroups and leave-one-state-out
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
rows = []
for grp, nice in [("by_state","State"),("by_race","Race and ethnicity"),("by_days_since_activation","Days since activation"),("by_activating_discipline","Discipline of contact")]:
    for k, v in subs[grp].items():
        if "auroc" in v["full"]: rows.append((f"{nice}: {k.replace('_',' to ') if grp=='by_days_since_activation' else NICE.get(k,k)} (n={v['full']['n']:,})", v["structured_history"]["auroc"], v["full"]["auroc"]))
yy = np.arange(len(rows))[::-1]
axs[0].scatter([r[1] for r in rows], yy, color=COL["M2_structured_history"], s=22, label=LABEL["M2_structured_history"], zorder=3); axs[0].scatter([r[2] for r in rows], yy, color=COL[FULL], s=22, marker="D", label=LABEL[FULL], zorder=3)
for i, r in zip(yy, rows): axs[0].plot([r[1], r[2]], [i, i], c="#888", lw=0.8)
axs[0].set_yticks(yy); axs[0].set_yticklabels([r[0] for r in rows], fontsize=7); axs[0].set(xlabel="AUROC, test era", title="a  Subgroup discrimination"); axs[0].legend(fontsize=6.5, frameon=False, loc="lower left"); axs[0].axvline(M["M2_structured_history"]["auroc"], c="#aaa", ls=":", lw=0.8)
lo = subs["leave_one_state_out"]; names = [k.replace("test_","").title() for k in lo]; x = np.arange(len(names))
axs[1].bar(x-0.18, [v["structured_history"]["auprc"] for v in lo.values()], 0.36, color=COL["M2_structured_history"], label="Structured + history"); axs[1].bar(x+0.18, [v["full_text"]["auprc"] for v in lo.values()], 0.36, color=COL[FULL], label="+ note text")
for i, v in enumerate(lo.values()): axs[1].plot([i-0.36, i+0.36], [v["event_rate"]]*2, c="k", ls=":", lw=0.9)
axs[1].set_xticks(x); axs[1].set_xticklabels([f"Held-out\n{n}\n(n={v['n']:,})" for n, v in zip(names, lo.values())], fontsize=7.5); axs[1].set(ylabel="AUPRC (dotted = event rate)", title="b  Leave-one-state-out"); axs[1].legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig(FIG/"figS1_subgroups_loso.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ---------------- Figure 5: narrative patterns
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
lx = {k.replace("lex_","").replace("_"," "): v for k, v in subs["lexicon_rates_test_era"].items() if v["rate_if_present"] is not None}; ks = sorted(lx, key=lambda k: lx[k]["rate_if_present"])
axs[0].barh(ks, [100*lx[k]["rate_if_absent"] for k in ks], color="#cfd8dc", label="Pattern absent"); axs[0].barh(ks, [100*lx[k]["rate_if_present"] for k in ks], height=0.5, color=COL[FULL], label="Pattern present in current note")
for i, k in enumerate(ks): axs[0].text(100*max(lx[k]["rate_if_present"], lx[k]["rate_if_absent"])+0.6, i, f"{100*lx[k]['prevalence']:.1f}% of notes", va="center", fontsize=6.5, color="#444")
axs[0].set_ylim(-0.6, len(ks)+0.9)
axs[0].set(xlabel="Disengagement within 90 days, % of contacts (test era)", title="a  Pre-specified engagement lexicon", xlim=(0, 60)); axs[0].legend(fontsize=6.5, frameon=False, loc="upper right", ncol=2)
pi = subs["permutation_importance_top25_auprc"][:15][::-1]; axs[1].barh([FEAT.get(p["feature"], p["feature"]) for p in pi], [p["drop_in_auprc"] for p in pi], color=COL["M2_structured_history"]); axs[1].set(xlabel="Drop in AUPRC when permuted", title="b  Permutation importance (full model without embeddings)"); axs[1].tick_params(axis="y", labelsize=7)
fig.tight_layout(); fig.savefig(FIG/"fig4_narrative_patterns.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ---------------- Figure 6: lever L1
L1 = lev["exposures"]["L1_inperson"]; fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
bins = L1["gate1_pretrend"]["bins"]; bk = list(bins); axs[0].errorbar(range(len(bk)), [bins[b]["diff_contacts_per_30d"] for b in bk], yerr=[[bins[b]["diff_contacts_per_30d"]-bins[b]["ci_95"][0] for b in bk],[bins[b]["ci_95"][1]-bins[b]["diff_contacts_per_30d"] for b in bk]], fmt="o", color=COL["M2_structured_history"], capsize=3)
axs[0].axhline(0, c="k", lw=0.8); axs[0].axhspan(-lev["equivalence_margin_abs"]*10, lev["equivalence_margin_abs"]*10, color="#eee", zorder=0); axs[0].set_xticks(range(len(bk))); axs[0].set_xticklabels([b.replace("pre","Pre-period bin ") for b in bk]); axs[0].set(ylabel="In-person minus remote:\ncontacts per 30 days before t", title=f"a  Pre-trend gate ({'passed' if L1['gate1_pretrend']['passed'] else 'failed'})")
ests = [("Within-patient conditional logit", L1["within_patient_fe"]["odds_ratio"], L1["within_patient_fe"]["ci_95"]), ("Marginal structural model (IPTW)", L1["msm_iptw"]["odds_ratio"], L1["msm_iptw"]["ci_95"])]
for i, (n, o, ci) in enumerate(ests): axs[1].errorbar(o, i, xerr=[[o-ci[0]],[ci[1]-o]], fmt="s", color=COL[FULL], capsize=3)
axs[1].axvline(1, c="k", ls=":", lw=0.8); axs[1].set_yticks(range(len(ests))); axs[1].set_yticklabels([e[0] for e in ests], fontsize=8); axs[1].set(xlabel="Odds ratio, disengagement in days 31 to 120\n(in-person vs remote contact at t)", title="b  Effect estimates within the equipoise region", xscale="log", xlim=(0.5, 2.2)); axs[1].set_ylim(-0.7, 1.7); axs[1].set_xticks([0.5, 0.75, 1.0, 1.5, 2.0]); axs[1].set_xticklabels(["0.5","0.75","1.0","1.5","2.0"]); axs[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.tight_layout(); fig.savefig(FIG/"figS2_lever_inperson.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ---------------- Tables 2-5 (markdown)
def ci(c, k): return f"{c[k]['diff']:+.3f} ({c[k]['ci_95'][0]:.3f} to {c[k]['ci_95'][1]:.3f})"
with open(TAB/"table2_model_performance.md", "w") as f:
    f.write(f"**Table 2. Prediction of disengagement within 90 days on the test era ({models['n_test']:,} decision points; {models['n_test_patients']:,} patients; {models['test_event_rate']*100:.1f}% event rate).** Net benefit is per contact at the flagging threshold; PPV is the positive predictive value; disengagements per 100 flags equals 100 times PPV.\n\n| Model | AUROC | AUPRC | Brier | Calibration slope | Net benefit, 20% flagged | PPV, 20% flagged | Sensitivity, 20% flagged | Disengagements per 100 flags (20%) |\n|---|---|---|---|---|---|---|---|---|\n")
    for m in ["M0_signal_risk","M1_structured","M2_structured_history","M3_plus_tfidf_text","M3b_plus_lexicon_tags","M4_plus_embeddings","M5_full","text_only_tfidf","embeddings_only"]:
        if m in M: v = M[m]; f.write(f"| {LABEL[m]} | {v['auroc']:.3f} | {v['auprc']:.3f} | {v['brier']:.3f} | {v['calibration_slope']:.2f} | {v['nb_flag20']:.4f} | {v['ppv_flag20']:.3f} | {v['sens_flag20']:.3f} | {100*v['ppv_flag20']:.1f} |\n")
    f.write("\n**Pre-specified contrasts (difference, 95% patient-cluster bootstrap interval, 1,000 resamples).**\n\n| Contrast | AUPRC | AUROC | Net benefit, 20% flagged | PPV, 20% flagged |\n|---|---|---|---|---|\n")
    for k, lab in [("full_vs_structured_history", f"{LABEL[FULL]} vs {LABEL['M2_structured_history']} (primary)"), ("text_vs_structured_history", f"{LABEL['M3_plus_tfidf_text']} vs {LABEL['M2_structured_history']}"), ("structured_history_vs_signal", f"{LABEL['M2_structured_history']} vs {LABEL['M0_signal_risk']} (secondary)"), ("full_vs_structured", f"{LABEL[FULL]} vs {LABEL['M1_structured']}")]:
        f.write(f"| {lab} | {ci(C[k],'auprc')} | {ci(C[k],'auroc')} | {ci(C[k],'nb_flag20')} | {ci(C[k],'ppv_flag20')} |\n")
    f.write("\n**Secondary outcomes (test era).**\n\n| Outcome | Event rate | Model | AUROC | AUPRC | Net benefit, 20% flagged |\n|---|---|---|---|---|---|\n")
    for yc, lab in [("y_explicit","Explicit exit status within 90 days"),("y_silent","Silent loss within 90 days"),("y_30","No contact within 30 days")]:
        b = models["secondary_outcomes"][yc]
        for m in ["M2_structured_history", FULL]:
            if m in b: f.write(f"| {lab} | {100*b['test_event_rate']:.1f}% | {LABEL[m]} | {b[m]['auroc']:.3f} | {b[m]['auprc']:.3f} | {b[m]['nb_flag20']:.4f} |\n")
with open(TAB/"table3_subgroups_loso.md", "w") as f:
    f.write("**Table 3. Discrimination by subgroup on the test era and leave-one-state-out validation.** Gains are full model minus structured plus contact history model.\n\n| Subgroup | n | Events | AUROC, structured + history | AUROC, full | AUROC gain | AUPRC, structured + history | AUPRC, full |\n|---|---|---|---|---|---|---|---|\n")
    for grp, nice in [("by_state","State"),("by_race","Race and ethnicity"),("by_days_since_activation","Days since activation"),("by_activating_discipline","Discipline of contact")]:
        for k, v in subs[grp].items():
            if "auroc" in v["full"]: f.write(f"| {nice}: {k.replace('_',' to ') if grp=='by_days_since_activation' else k} | {v['full']['n']:,} | {v['full']['events']:,} | {v['structured_history']['auroc']:.3f} | {v['full']['auroc']:.3f} | {v['auroc_gain']:+.3f} | {v['structured_history']['auprc']:.3f} | {v['full']['auprc']:.3f} |\n")
    f.write("\n| Leave-one-state-out (train on two states, test on the third; all eligible decision points) | n | Event rate | AUROC, structured + history | AUROC, + note text | AUPRC, structured + history | AUPRC, + note text |\n|---|---|---|---|---|---|---|\n")
    for k, v in subs["leave_one_state_out"].items(): f.write(f"| Held-out {k.replace('test_','').title()} | {v['n']:,} | {100*v['event_rate']:.1f}% | {v['structured_history']['auroc']:.3f} | {v['full_text']['auroc']:.3f} | {v['structured_history']['auprc']:.3f} | {v['full_text']['auprc']:.3f} |\n")
with open(TAB/"tableS16_levers.md", "w") as f:
    f.write(f"**Supplementary Table S16. Care-team actions at the contact and disengagement in days 31 to 120 (lever analysis; {lev['n_lever_eligible']:,} eligible decision points; base rate {100*lev['base_rate']:.1f}%).** Equipoise region: estimated propensity between 0.10 and 0.90. Gate 1 requires flat pre-period engagement (each of three 30-day bins within {100*lev['equivalence_margin_abs']/lev['base_rate']:.0f}% of the base rate with an interval covering zero); Gate 2 requires agreement between the within-patient and marginal structural estimators.\n\n| Exposure at t | Prevalence | Decision points in equipoise | Informative patients (switchers) | Gate 1 | Within-patient OR (95% CI) | MSM OR (95% CI) | Gate 2 | Risk-difference bounds, gamma 1.25 |\n|---|---|---|---|---|---|---|---|---|\n")
    for k, v in lev["exposures"].items():
        nm = {"L1_inperson":"In-person vs remote contact","L2_pharm":"Pharmacist review vs not","L3_therapy":"Therapy session vs not","L4_disc_change":"Different discipline from previous contact"}[k]
        if "within_patient_fe" in v: f.write(f"| {nm} | {100*v['prevalence']:.1f}% | {v['equipoise_dps']:,} | {v['informative_patients']} | {'Pass' if v['gate1_pretrend']['passed'] else 'Fail'} | {v['within_patient_fe']['odds_ratio']:.2f} ({v['within_patient_fe']['ci_95'][0]:.2f} to {v['within_patient_fe']['ci_95'][1]:.2f}) | {v['msm_iptw']['odds_ratio']:.2f} ({v['msm_iptw']['ci_95'][0]:.2f} to {v['msm_iptw']['ci_95'][1]:.2f}) | {'Agree' if v['gate2_agreement'] else 'Disagree'} | {v['sensitivity_bounds_risk_difference']['gamma_1.25'][0]:+.3f} to {v['sensitivity_bounds_risk_difference']['gamma_1.25'][1]:+.3f} |\n")
        else: f.write(f"| {nm} | {100*v['prevalence']:.1f}% | {v['equipoise_dps']:,} | not estimable | not estimable | not estimable | not estimable | not estimable | not estimable |\n")
with open(TAB/"table5_sequence_models.md", "w") as f:
    if seq:
        f.write(f"**Table 5. Supplementary architecture comparison on the test era ({seq['n_test']:,} decision points).** Sequence models read the last {seq['sequence_length']} contacts (token dimension {seq['token_dim']}: discipline, modality, timing, and 32 principal components of the note embedding) plus the same nine structured covariates; three seeds, early stopping on a patient-grouped validation fold; predictions averaged over seeds. Parentheses are 95% member-bootstrap intervals (1,000 resamples).\n\n| Model | Parameters | AUROC (95% CI) | AUPRC (95% CI) | AUPRC by seed |\n|---|---|---|---|---|\n")
        MCq = J("metric_cis.json") or {"models": {}}
        for k, v in seq["models"].items():
            c = MCq["models"].get("seq_" + k, {}); ca = c.get("auroc", {}).get("ci_95"); cp = c.get("auprc", {}).get("ci_95")
            f.write(f"| {k.replace('_',' ')} | {v['params']:,} | {v['auroc']:.3f}{f' ({ca[0]:.3f} to {ca[1]:.3f})' if ca else ''} | {v['auprc']:.3f}{f' ({cp[0]:.3f} to {cp[1]:.3f})' if cp else ''} | {', '.join(f'{s:.3f}' for s in v['seed_auprc'])} |\n")
        for k, v in seq["reference_gbm"].items(): f.write(f"| Reference: {LABEL.get(k,k)} (gradient-boosted trees) | | {v['auroc']:.3f} | {v['auprc']:.3f} | |\n")
    else: f.write("Sequence model results not yet available.\n")
print("canonical.json written; figures:", sorted(p.name for p in FIG.glob('*.png')), "tables:", sorted(p.name for p in TAB.glob('*.md')))
print(json.dumps(canon["derived"], indent=1, default=str)[:3000])

# ---------------- Supplementary tables S3 (sensitivity) and S4 (lexicon)
if sens:
    NM = {"base_eligible60_activation_split": "Base specification (at least 60 enrolled days; activation-cohort split)", "eligibility_90_enrolled_days": "Eligibility: 90 enrolled days in the following 90 days", "no_eligibility_restriction_enrollment_covariate": "No eligibility restriction; enrolled days as a covariate", "training_contacts_before_2025_07_01_only": "Training contacts restricted to dates before 2025-07-01", "outcome_y_30": "Outcome: no contact within 30 days", "outcome_y_silent": "Outcome: silent loss within 90 days", "outcome_y_explicit": "Outcome: explicit exit status within 90 days"}
    with open(TAB/"tableS3_sensitivity.md", "w") as f:
        f.write("**Supplementary Table S3. Sensitivity of the structured plus contact history model (M2) and the model with note text (M3, TF-IDF stack) to eligibility, split, and outcome definition.** Each row refits both models and evaluates on the test era under the stated specification; embeddings, lexicon, and tags are not included in M3. Parentheses are 95% member-bootstrap intervals (300 resamples).\n\n| Specification | Training decision points | Test decision points | Test event rate | AUROC, M2 | AUPRC, M2 | AUROC, M3 | AUPRC, M3 | AUPRC gain from text |\n|---|---|---|---|---|---|---|---|---|\n")
        def cis(v, m, k):
            c = v.get("ci", {}).get(m, {}).get(k); return f" ({c['ci_95'][0]:.3f} to {c['ci_95'][1]:.3f})" if c and c.get("ci_95") else ""
        for k, v in sens.items():
            g = v.get("auprc_gain_text_ci_95"); gs = f" ({g[0]:+.3f} to {g[1]:+.3f})" if g else ""
            f.write(f"| {NM.get(k,k)} | {v['n_train']:,} | {v['n_test']:,} | {100*v['test_event_rate']:.1f}% | {v['M2_structured_history']['auroc']:.3f}{cis(v,'M2_structured_history','auroc')} | {v['M2_structured_history']['auprc']:.3f}{cis(v,'M2_structured_history','auprc')} | {v['M3_plus_tfidf_text']['auroc']:.3f}{cis(v,'M3_plus_tfidf_text','auroc')} | {v['M3_plus_tfidf_text']['auprc']:.3f}{cis(v,'M3_plus_tfidf_text','auprc')} | {v['auprc_gain_text']:+.3f}{gs} |\n")
    canon["derived"]["sensitivity_auprc_gain_range"] = [min(v["auprc_gain_text"] for v in sens.values()), max(v["auprc_gain_text"] for v in sens.values())]
    canon["derived"]["sensitivity_m2_auroc_range_primary"] = [min(v["M2_structured_history"]["auroc"] for k, v in sens.items() if not k.startswith("outcome")), max(v["M2_structured_history"]["auroc"] for k, v in sens.items() if not k.startswith("outcome"))]
    json.dump(canon, open(R/"canonical.json", "w"), indent=1, default=str)
with open(TAB/"tableS4_lexicon.md", "w") as f:
    f.write("**Supplementary Table S4. Pre-specified engagement lexicon (regular expressions applied to lower-cased note text) and prevalence in the current note across all eligible decision points.**\n\n| Pattern | Regular expression | Prevalence in current note |\n|---|---|---|\n")
    for k, v in fmeta["lexicon_patterns"].items(): f.write(f"| {k.replace('lex_','').replace('_',' ')} | `{v}` | {100*fmeta['lexicon_prevalence_current_note'][k]:.1f}% |\n")
print("supplementary tables written")

# ======================= Amendment v3.1 outputs: landmark, simple score, capacity, fairness, additional levers, clinical stakes =======================
LM, SS, CAP, FAIR, LV2, CS, PHM = J("landmark.json"), J("simple_score.json"), J("capacity.json"), J("fairness.json"), J("levers2.json"), J("clinical_stakes.json"), J("physician_sample_meta.json")
canon.update({"landmark": LM, "simple_score": SS, "capacity": CAP, "fairness": FAIR, "levers2": LV2, "clinical_stakes": CS, "physician_sample": PHM})
canon["generated_from"] += ["landmark.json","simple_score.json","capacity.json","fairness.json","levers2.json","clinical_stakes.json","physician_sample_meta.json"]
if LM and LM.get("headline_landmark"):
    H = LM["landmarks"][LM["headline_landmark"]]; ov = H["overall"]
    canon["derived"]["landmark_headline"] = {"date": LM["headline_landmark"], "train_n": H["train"]["n"], "train_events": H["train"]["events"], "scored_n": ov["n"], "scored_events": ov["events"], "auroc": ov["auroc"], "auprc": ov["auprc"], "calibration_slope": ov["calibration_slope"], "calibration_intercept": ov["calibration_intercept"], "flag_rate": ov["flag_rate"], "per_100_flags": ov["per_100_flags"], "sensitivity_at_threshold": ov["sensitivity_at_threshold"],
        "unseen_members_auroc": H["members_not_seen_in_training"].get("auroc"), "unseen_members_auprc": H["members_not_seen_in_training"].get("auprc"), "unseen_n": H["members_not_seen_in_training"].get("n"), "seen_members_auroc": H["members_seen_in_training"].get("auroc"), "seen_n": H["members_seen_in_training"].get("n"),
        "months_available": len(H["by_month_since_landmark"]), "auroc_by_month": {k: v.get("auroc") for k, v in H["by_month_since_landmark"].items()}, "top_decile_observed": H["by_training_decile"]["10"]["observed_rate"], "bottom_decile_observed": H["by_training_decile"]["1"]["observed_rate"], "ci_95": LM.get("headline_ci_95")}
    canon["derived"]["landmark_auroc_range_all"] = [min(v["overall"]["auroc"] for v in LM["landmarks"].values()), max(v["overall"]["auroc"] for v in LM["landmarks"].values())]
    canon["derived"]["landmark_auprc_range_all"] = [min(v["overall"]["auprc"] for v in LM["landmarks"].values()), max(v["overall"]["auprc"] for v in LM["landmarks"].values())]
    canon["derived"]["landmark_intercept_range_all"] = [min(v["overall"]["calibration_intercept"] for v in LM["landmarks"].values()), max(v["overall"]["calibration_intercept"] for v in LM["landmarks"].values())]
    canon["derived"]["landmark_n"] = len(LM["landmarks"])
    # Figure 7: rolling landmark
    ks = sorted(LM["landmarks"]); fig, axs = plt.subplots(1, 3, figsize=(12, 4.8))
    axs[0].plot(range(len(ks)), [LM["landmarks"][k]["overall"]["auroc"] for k in ks], "o-", color=COL[FULL], label="AUROC"); axs[0].plot(range(len(ks)), [LM["landmarks"][k]["overall"]["auprc"] for k in ks], "s-", color=COL["M2_structured_history"], label="AUPRC")
    axs[0].axvline(ks.index(LM["headline_landmark"]), c="#aaa", ls=":", lw=1); axs[0].set_xticks(range(len(ks))); axs[0].set_xticklabels([k[:7] for k in ks], rotation=60, fontsize=7); axs[0].set(ylabel="Performance on all contacts after the landmark", title="a  Frozen model by landmark", ylim=(0.5, 1.0)); axs[0].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.3, -0.42), ncol=2)
    ax2 = axs[0].twinx(); ax2.plot(range(len(ks)), [LM["landmarks"][k]["overall"]["calibration_intercept"] for k in ks], "^--", color="#888", lw=1, label="Calibration intercept"); ax2.set_ylabel("Calibration intercept", color="#888"); ax2.axhline(0, c="#ccc", lw=0.6); ax2.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.8, -0.42), ncol=1)
    bm = H["by_month_since_landmark"]; ms = [k for k in bm if "auroc" in bm[k]]
    axs[1].plot([int(k) for k in ms], [bm[k]["auroc"] for k in ms], "o-", color=COL[FULL], label="AUROC"); axs[1].plot([int(k) for k in ms], [bm[k]["event_rate"] for k in ms], "s--", color="#888", label="Event rate")
    axs[1].plot([int(k) for k in ms], [bm[k]["per_100_flags"]/100 for k in ms], "d-", color=COL["M2_structured_history"], label="Disengagements per flag"); axs[1].set(xlabel=f"Months after landmark {LM['headline_landmark']}", title="b  Decay after the headline landmark", ylim=(0, 1)); axs[1].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    dd = H["by_training_decile"]; axs[2].bar([int(k) for k in dd], [100*(dd[k]["observed_rate"] or 0) for k in dd], color=COL[FULL]); axs[2].axhline(100*ov["event_rate"], c="k", ls=":", lw=0.8); axs[2].set(xlabel="Decile of predicted risk (frozen cutpoints)", ylabel="Observed disengagement, %", title="c  Realized disengagement by frozen decile")
    fig.tight_layout(); fig.savefig(FIG/"fig5_landmark_validation.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    with open(TAB/"tableS5_landmark.md", "w") as f:
        f.write("**Supplementary Table S5. Frozen-model rolling-landmark validation.** At each month-end landmark the full model without embeddings is trained on eligible decision points dated at least 90 days before the landmark and scores every eligible decision point after it; the flagging threshold is the 80th percentile of training predictions. Parentheses are 95% member-bootstrap intervals (300 resamples).\n\n| Landmark | Training decision points (events) | Scored decision points (events) | AUROC | AUPRC | Calibration slope | Calibration intercept | Flag rate | Disengagements per 100 flags | Sensitivity at threshold | AUROC, members not in training |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for k in ks:
            v = LM["landmarks"][k]; o = v["overall"]; c = v.get("overall_ci", {})
            def w(key, val, d=3, pct=False, sign=False):
                x = c.get(key, {}).get("ci_95"); fmt = (lambda z: f"{100*z:.1f}%") if pct else (lambda z: f"{z:+.{d}f}" if sign else f"{z:.{d}f}"); return fmt(val) + (f" ({fmt(x[0])} to {fmt(x[1])})" if x else "")
            f.write(f"| {k}{' (headline)' if k == LM['headline_landmark'] else ''} | {v['train']['n']:,} ({v['train']['events']:,}) | {o['n']:,} ({o['events']:,}) | {w('auroc', o['auroc'])} | {w('auprc', o['auprc'])} | {w('calibration_slope', o['calibration_slope'], 2)} | {w('calibration_intercept', o['calibration_intercept'], 2, sign=True)} | {w('flag_rate', o['flag_rate'], pct=True)} | {w('per_100_flags', o['per_100_flags'], 1)} | {w('sensitivity', o['sensitivity_at_threshold'])} | {v['members_not_seen_in_training'].get('auroc', float('nan')):.3f} |\n")
        f.write(f"\n**Headline landmark {LM['headline_landmark']}, by month since landmark.**\n\n| Month | Scored decision points | Event rate | AUROC | AUPRC | Calibration slope | Disengagements per 100 flags |\n|---|---|---|---|---|---|---|\n")
        for k in ms: v = bm[k]; f.write(f"| {k} | {v['n']:,} | {100*v['event_rate']:.1f}% | {v['auroc']:.3f} | {v['auprc']:.3f} | {v['calibration_slope']:.2f} | {v['per_100_flags']:.1f} |\n")
if SS:
    canon["derived"]["simple_score_summary"] = {"auroc": SS["points_score"]["auroc"], "auprc": SS["points_score"]["auprc"], "per_100_flags_20pct": SS["points_score"]["at_20pct_flagged"]["per_100_flags"], "sensitivity_20pct": SS["points_score"]["at_20pct_flagged"]["sensitivity"], "share_of_gain_pct": round(100*SS["share_of_full_model_gain_over_M0_captured"]["points_score"], 1), "n_predictors": len(SS["predictors"])}
    with open(TAB/"tableS6_simple_score.md", "w") as f:
        f.write(f"**Supplementary Table S6. Eight-predictor points score (training era fit; test era evaluation, {SS['n_test']:,} decision points).** Points equal the logistic coefficient scaled so the largest is 10; log-transformed counts contribute their log value times the points.\n\n| Predictor | Coefficient | Points |\n|---|---|---|\n")
        for k in SS["predictors"]: f.write(f"| {k.replace('_',' ')} | {SS['coefficients'][k]:+.3f} | {SS['points'][k]:+d} |\n")
        def sc(m, k, d=3):
            c = SS.get("ci", {}).get(m, {}).get(k); return f"{c['estimate']:.{d}f} ({c['ci_95'][0]:.{d}f} to {c['ci_95'][1]:.{d}f})" if c and c.get("ci_95") else ""
        f.write(f"\n| Model | AUROC (95% CI) | AUPRC (95% CI) | Disengagements per 100 flags at 20% (95% CI) | Sensitivity at 20% (95% CI) | Share of full-model AUPRC gain over M0 captured |\n|---|---|---|---|---|---|\n| Logistic (8 predictors) | {sc('logistic','auroc')} | {sc('logistic','auprc')} | {sc('logistic','per_100_flags',1)} | {sc('logistic','sensitivity')} | {100*SS['share_of_full_model_gain_over_M0_captured']['logistic']:.1f}% |\n| Integer points score | {sc('points_score','auroc')} | {sc('points_score','auprc')} | {sc('points_score','per_100_flags',1)} | {sc('points_score','sensitivity')} | {100*SS['share_of_full_model_gain_over_M0_captured']['points_score']:.1f}% |\n")
if CAP:
    with open(TAB/"tableS7_capacity.md", "w") as f:
        f.write(f"**Supplementary Table S7. True disengagements reached per care team per week at fixed re-engagement capacity (test era; {CAP['team_weeks']:,} team-weeks; mean {CAP['mean_contacts_per_team_week']} contacts and {CAP['mean_disengagements_per_team_week']} disengagements per team-week).** Parentheses are 95% bootstrap intervals over team-weeks (1,000 resamples).\n\n| Weekly attempts per team | Full model | Structured + history | Deployed risk score | All first-30-day contacts | Random |\n|---|---|---|---|---|---|\n")
        def cc(v, k): x = v.get(k + "_ci_95"); return f"{v[k]:.2f}" + (f" ({x[0]:.2f} to {x[1]:.2f})" if x else "")
        for cap, v in CAP["capacity"].items(): f.write(f"| {cap} | {cc(v,'full_model')} | {cc(v,'structured_history')} | {cc(v,'deployed_risk_score')} | {cc(v,'first_30_days')} | {cc(v,'random')} |\n")
    canon["derived"]["capacity_10"] = CAP["capacity"]["10"]
if FAIR:
    with open(TAB/"tableS8_fairness.md", "w") as f:
        f.write("**Supplementary Table S8. Full-model performance by subgroup at the single overall 20% threshold (test era).** Parentheses are 95% member-bootstrap intervals within each group (300 resamples); differences versus the reference group carry 95% member-bootstrap intervals.\n\n| Dimension | Group | n | Event rate | AUROC | AUPRC | Calibration slope | Calibration intercept | Flag rate | Sensitivity | PPV | Sensitivity difference vs reference (95% CI) | PPV difference vs reference (95% CI) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for dim, blk in FAIR["groups"].items():
            for lev, v in blk["levels"].items():
                if "note" in v: f.write(f"| {dim.replace('_',' ')} | {lev} | {v['n']:,} | insufficient | | | | | | | | | |\n"); continue
                m = v["M5_full"]; sd = m.get("sensitivity_diff_vs_ref_ci95"); pd_ = m.get("ppv_diff_vs_ref_ci95")
                def fc(k, val, d=3, sign=False):
                    x = m.get("ci", {}).get(k, {}).get("ci_95"); fmt = (lambda z: f"{z:+.{d}f}") if sign else (lambda z: f"{z:.{d}f}"); return fmt(val) + (f" ({fmt(x[0])} to {fmt(x[1])})" if x else "")
                f.write(f"| {dim.replace('_',' ')} | {lev}{' (reference)' if lev == blk['reference'] else ''} | {v['n']:,} | {100*v['event_rate']:.1f}% | {fc('auroc', m['auroc'])} | {fc('auprc', m['auprc'])} | {fc('calibration_slope', m['calibration_slope'], 2)} | {fc('calibration_intercept', m['calibration_intercept'], 2, True)} | {100*m['flag_rate']:.1f}% | {fc('sensitivity', m['sensitivity'])} | {fc('ppv', m['ppv'])} | {f'{sd[0]:+.3f} to {sd[1]:+.3f}' if sd else 'reference'} | {f'{pd_[0]:+.3f} to {pd_[1]:+.3f}' if pd_ else 'reference'} |\n")
    flat = [(dim, lev, v["M5_full"]) for dim, blk in FAIR["groups"].items() for lev, v in blk["levels"].items() if "note" not in v]
    canon["derived"]["fairness_auroc_range"] = [min(m["auroc"] for _, _, m in flat), max(m["auroc"] for _, _, m in flat)]; canon["derived"]["fairness_slope_range"] = [min(m["calibration_slope"] for _, _, m in flat), max(m["calibration_slope"] for _, _, m in flat)]
    canon["derived"]["fairness_flag_rate_range"] = [min(m["flag_rate"] for _, _, m in flat), max(m["flag_rate"] for _, _, m in flat)]; canon["derived"]["fairness_sens_range"] = [min(m["sensitivity"] for _, _, m in flat), max(m["sensitivity"] for _, _, m in flat)]; canon["derived"]["fairness_ppv_range"] = [min(m["ppv"] for _, _, m in flat), max(m["ppv"] for _, _, m in flat)]
    canon["derived"]["fairness_groups_with_sens_diff_excluding_zero"] = [f"{dim}:{lev}" for dim, lev, m in flat if m.get("sensitivity_diff_vs_ref_ci95") and (m["sensitivity_diff_vs_ref_ci95"][0] > 0 or m["sensitivity_diff_vs_ref_ci95"][1] < 0)]
    canon["derived"]["fairness_groups_with_ppv_diff_excluding_zero"] = [f"{dim}:{lev}" for dim, lev, m in flat if m.get("ppv_diff_vs_ref_ci95") and (m["ppv_diff_vs_ref_ci95"][0] > 0 or m["ppv_diff_vs_ref_ci95"][1] < 0)]
if LV2 or CS:
    with open(TAB/"tableS9_levers2_clinical_stakes.md", "w") as f:
        if LV2:
            f.write("**Supplementary Table S9a. Additional levers (outcome: no completed contact in days 31 to 120 after the next contact and no graduation by day 120).**\n\n| Exposure | Index decision points | Prevalence | Equipoise decision points | Within-member OR (95% CI) | MSM OR (95% CI) | MSM risk difference | Detectable OR at 80% power (MSM) | Gate 1 | Gate 2 | Risk-difference bounds, gamma 1.25 |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
            for k, nm in [("L5_second_contact_within_7_days","Second contact within 7 days of the first contact"),("L6_same_person_continuity","Next contact by the same care team member")]:
                v = LV2[k]; fe = v.get("within_member_fe", {}); ms = v.get("msm_iptw"); g1 = v.get("gate1_pretrend", {}).get("passed"); b = v.get("sensitivity_bounds_risk_difference", {}).get("gamma_1.25")
                fe_s = f"{fe['odds_ratio']:.2f} ({fe['ci_95'][0]:.2f} to {fe['ci_95'][1]:.2f})" if "odds_ratio" in fe else ("not applicable" if k.startswith("L5") else "not estimable")
                ms_s = f"{ms['odds_ratio']:.2f} ({ms['ci_95'][0]:.2f} to {ms['ci_95'][1]:.2f})" if ms else "not estimable"; rd_s = f"{ms['risk_difference']:+.3f}" if ms else ""; mde_s = str(ms.get("mde_or_80pct_power", "")) if ms else ""
                g1_s = "Pass" if g1 else ("Fail" if g1 is False else "not applicable"); g2 = v.get("gate2_agreement"); g2_s = "Agree" if g2 else ("Disagree" if g2 is False else "not applicable"); b_s = f"{b[0]:+.3f} to {b[1]:+.3f}" if b else ""
                f.write(f"| {nm} | {v['n']:,} | {100*v['prevalence']:.1f}% | {v.get('equipoise_dps', 0):,} | {fe_s} | {ms_s} | {rd_s} | {mde_s} | {g1_s} | {g2_s} | {b_s} |\n")
        if CS:
            f.write(f"\n**Supplementary Table S9b. Acute care events in days 1 to 90 after test-era decision points ({CS['n_test_dps']:,} decision points; {CS['members']:,} members), disengaged versus retained.** Events per 100 decision points; incidence rate ratios from Poisson regression with member-clustered standard errors, adjusted for risk percentile, days since activation, prior acute events, age, behavioral health and substance use flags, state, and discipline.\n\n| Outcome | Disengaged (n = {CS['rates']['y1']['n']:,}) | Retained (n = {CS['rates']['y0']['n']:,}) | Unadjusted IRR (95% CI) | Adjusted IRR (95% CI) |\n|---|---|---|---|---|\n")
            for col, nm, key in [("acute90","ED visits and inpatient admissions","acute"),("ed90","ED visits","ed"),("ip90","Inpatient admissions","inpatient")]:
                r = CS[f"irr_{col}"]; f.write(f"| {nm} | {CS['rates']['y1'][key]:.2f} | {CS['rates']['y0'][key]:.2f} | {r['unadjusted']:.2f} ({r['unadjusted_ci_95'][0]:.2f} to {r['unadjusted_ci_95'][1]:.2f}) | {r['adjusted']:.2f} ({r['adjusted_ci_95'][0]:.2f} to {r['adjusted_ci_95'][1]:.2f}) |\n")
            f.write(f"| Any acute event, % of decision points | {CS['rates']['y1']['any_acute_pct']:.1f}% | {CS['rates']['y0']['any_acute_pct']:.1f}% | | |\n")
json.dump(canon, open(R/"canonical.json", "w"), indent=1, default=str); print("amendment outputs rendered")

# ======================= Metric intervals, ablation, quarterly refit =======================
MC, AB, RF = J("metric_cis.json"), J("ablation.json"), J("rolling_refit.json")
canon.update({"metric_cis": MC, "ablation": AB, "rolling_refit": RF}); canon["generated_from"] += ["metric_cis.json","ablation.json","rolling_refit.json"]
if MC:
    with open(TAB/"table2_model_performance.md", "w") as f:
        f.write(f"**Table 2. Prediction of disengagement within 90 days on the test era ({models['n_test']:,} decision points; {models['n_test_patients']:,} members; {models['test_event_rate']*100:.1f}% event rate).** Operating-point metrics use each model's own 80th-percentile threshold (20% of contacts flagged). Intervals are 95% percentile intervals from 1,000 bootstrap resamples of members. Disengagements per 100 flags equals 100 times the positive predictive value (PPV); NPV, negative predictive value.\n\n| Model | AUROC (95% CI) | AUPRC (95% CI) | Brier | Calibration slope (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | PPV (95% CI) | NPV (95% CI) | F1 (95% CI) | Net benefit (95% CI) |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        def c(v, k, d=3): return f"{v[k]['estimate']:.{d}f} ({v[k]['ci_95'][0]:.{d}f} to {v[k]['ci_95'][1]:.{d}f})"
        for m in ["M0_signal_risk","M1_structured","M2_structured_history","M3_plus_tfidf_text","M3b_plus_lexicon_tags","M4_plus_embeddings","M5_full","text_only_tfidf","embeddings_only","seq_GRU","seq_Transformer","seq_Mamba_SSM"]:
            if m in MC["models"]:
                v = MC["models"][m]; lab = LABEL.get(m, m.replace("seq_", "Sequence model: ").replace("_SSM", " (selective state-space)"))
                f.write(f"| {lab} | {c(v,'auroc')} | {c(v,'auprc')} | {v['brier']['estimate']:.3f} | {c(v,'calibration_slope',2)} | {c(v,'sensitivity')} | {c(v,'specificity')} | {c(v,'ppv')} | {c(v,'npv')} | {c(v,'f1')} | {c(v,'net_benefit',4)} |\n")
        f.write("\n**Pre-specified contrasts (difference, 95% member-bootstrap interval, 1,000 resamples).**\n\n| Contrast | AUPRC | AUROC | Net benefit, 20% flagged | PPV, 20% flagged |\n|---|---|---|---|---|\n")
        for k, lab in [("full_vs_structured_history", f"{LABEL[FULL]} vs {LABEL['M2_structured_history']} (primary)"), ("text_vs_structured_history", f"{LABEL['M3_plus_tfidf_text']} vs {LABEL['M2_structured_history']}"), ("structured_history_vs_signal", f"{LABEL['M2_structured_history']} vs {LABEL['M0_signal_risk']} (secondary)"), ("full_vs_structured", f"{LABEL[FULL]} vs {LABEL['M1_structured']}")]:
            f.write(f"| {lab} | {ci(C[k],'auprc')} | {ci(C[k],'auroc')} | {ci(C[k],'nb_flag20')} | {ci(C[k],'ppv_flag20')} |\n")
        f.write("\n**Secondary outcomes (test era).**\n\n| Outcome | Event rate | Model | AUROC | AUPRC | Net benefit, 20% flagged |\n|---|---|---|---|---|---|\n")
        for yc, lab in [("y_explicit","Explicit exit status within 90 days"),("y_silent","Silent loss within 90 days"),("y_30","No contact within 30 days")]:
            b = models["secondary_outcomes"][yc]
            for m in ["M2_structured_history", FULL]:
                if m in b: f.write(f"| {lab} | {100*b['test_event_rate']:.1f}% | {LABEL[m]} | {b[m]['auroc']:.3f} | {b[m]['auprc']:.3f} | {b[m]['nb_flag20']:.4f} |\n")
        f.write(f"\nDecision-point (rather than member) bootstrap interval for the full model, for comparison of width: AUROC {MC['M5_full_decision_point_bootstrap']['auroc'][0]:.3f} to {MC['M5_full_decision_point_bootstrap']['auroc'][1]:.3f}; AUPRC {MC['M5_full_decision_point_bootstrap']['auprc'][0]:.3f} to {MC['M5_full_decision_point_bootstrap']['auprc'][1]:.3f}.\n")
    # Table 3 with subgroup intervals appended
    with open(TAB/"table3_subgroups_loso.md", "a") as f:
        f.write("\n**Subgroup intervals (95% member-bootstrap, 500 resamples within subgroup).**\n\n| Subgroup | n | AUROC, structured + history (95% CI) | AUROC, full (95% CI) | AUPRC, structured + history (95% CI) | AUPRC, full (95% CI) |\n|---|---|---|---|---|---|\n")
        for k, v in MC["subgroups"].items():
            a, b = v["M2_structured_history"], v["M5_full"]; f.write(f"| {k.replace(':', ': ').replace('state: ', 'State: ').replace('race: ', 'Race and ethnicity: ').replace('VIRGINIA','Virginia').replace('WASHINGTON','Washington').replace('OHIO','Ohio')} | {v['n']:,} | {a['auroc']:.3f} ({a['auroc_ci_95'][0]:.3f} to {a['auroc_ci_95'][1]:.3f}) | {b['auroc']:.3f} ({b['auroc_ci_95'][0]:.3f} to {b['auroc_ci_95'][1]:.3f}) | {a['auprc']:.3f} ({a['auprc_ci_95'][0]:.3f} to {a['auprc_ci_95'][1]:.3f}) | {b['auprc']:.3f} ({b['auprc_ci_95'][0]:.3f} to {b['auprc_ci_95'][1]:.3f}) |\n")
    canon["derived"]["ci"] = {m: {k: MC["models"][m][k] for k in ["auroc","auprc","sensitivity","specificity","ppv","npv","f1","calibration_slope","brier","net_benefit"]} for m in ["M0_signal_risk","M2_structured_history","M5_full"]}
if AB:
    with open(TAB/"tableS10_ablation.md", "w") as f:
        f.write(f"**Supplementary Table S10. Feature-group ablation and learner variants of the full model (test era, {models['n_test']:,} decision points).** Leave-one-group-out removes one predictor block from the full model; only-one-group fits the block alone. Logistic regression uses the same standardized features with L2 penalty (C = 0.1). Parentheses are 95% member-bootstrap intervals (300 resamples).\n\n| Specification | AUROC (95% CI) | AUPRC (95% CI) | Change in AUPRC from full model |\n|---|---|---|---|\n| Full model (gradient boosting) | {AB['full_model']['auroc']:.3f}{(lambda x: f' ({x[0]:.3f} to {x[1]:.3f})' if x else '')(AB['full_model'].get('auroc_ci_95'))} | {AB['full_model']['auprc']:.3f}{(lambda x: f' ({x[0]:.3f} to {x[1]:.3f})' if x else '')(AB['full_model'].get('auprc_ci_95'))} | |\n")
        NM = {"structured":"structured data","contact_history":"contact history","lexicon_and_tags":"lexicon and action tags","note_embeddings":"note embeddings","tfidf_note_stack":"TF-IDF note stack"}
        def ac(v, k): x = v.get(k + "_ci_95"); return f"{v[k]:.3f}" + (f" ({x[0]:.3f} to {x[1]:.3f})" if x else "")
        for g, v in AB["leave_one_group_out"].items(): f.write(f"| Without {NM[g]} | {ac(v,'auroc')} | {ac(v,'auprc')} | {-v['auprc_drop']:+.3f} |\n")
        for g, v in AB["only_one_group"].items(): f.write(f"| Only {NM[g]} | {ac(v,'auroc')} | {ac(v,'auprc')} | {v['auprc'] - AB['full_model']['auprc']:+.3f} |\n")
        v = AB["logistic_regression_same_features"]; f.write(f"| Logistic regression, same features | {ac(v,'auroc')} | {ac(v,'auprc')} | {v['auprc'] - AB['full_model']['auprc']:+.3f} |\n")
        for g, v in AB["hyperparameter_variants"].items(): f.write(f"| Gradient boosting, {g.replace('_',' ')} | {ac(v,'auroc')} | {ac(v,'auprc')} | {v['auprc'] - AB['full_model']['auprc']:+.3f} |\n")
    canon["derived"]["ablation_summary"] = {"largest_drop_group": max(AB["leave_one_group_out"].items(), key=lambda kv: kv[1]["auprc_drop"])[0], "drops": {g: v["auprc_drop"] for g, v in AB["leave_one_group_out"].items()}, "lr_auroc": AB["logistic_regression_same_features"]["auroc"], "lr_auprc": AB["logistic_regression_same_features"]["auprc"], "hp_auprc_range": [min(v["auprc"] for v in AB["hyperparameter_variants"].values()), max(v["auprc"] for v in AB["hyperparameter_variants"].values())]}
if RF:
    with open(TAB/"tableS11_quarterly_refit.md", "w") as f:
        def e(v, k, d=3): x = v[k]; return f"{x['estimate']:.{d}f} ({x['ci_95'][0]:.{d}f} to {x['ci_95'][1]:.{d}f})" if isinstance(x, dict) else str(x)
        f.write(f"**Supplementary Table S11. Quarterly refitting versus a single frozen model, pooled over decision points dated after 31 March 2025 ({RF['pooled_n']:,} decision points).** {RF['design']}. Intervals are 95% member-bootstrap (500 resamples).\n\n| Deployment | AUROC | AUPRC | Calibration slope | Calibration intercept | Flag rate | Disengagements per 100 flags | Sensitivity |\n|---|---|---|---|---|---|---|---|\n")
        for k, lab in [("refit_quarterly","Refit each quarter"),("frozen_2025_03_31","Frozen at 2025-03-31")]:
            v = RF[k]; f.write(f"| {lab} | {e(v,'auroc')} | {e(v,'auprc')} | {e(v,'calibration_slope',2)} | {e(v,'calibration_intercept',2)} | {e(v,'flag_rate')} | {e(v,'per_100_flags',1)} | {e(v,'sensitivity')} |\n")
        v = RF["frozen_2025_11_30_with_ci"]; f.write(f"| Frozen at 2025-11-30 (headline landmark; decision points after that date only) | {e(v,'auroc')} | {e(v,'auprc')} | {e(v,'calibration_slope',2)} | {e(v,'calibration_intercept',2)} | {e(v,'flag_rate')} | {e(v,'per_100_flags',1)} | {e(v,'sensitivity')} |\n")
        f.write("\n**By quarter.**\n\n| Quarter starting | Decision points | Refit: AUROC | Refit: calibration intercept | Refit: per 100 flags | Frozen: AUROC | Frozen: calibration intercept | Frozen: per 100 flags |\n|---|---|---|---|---|---|---|---|\n")
        for q, v in RF["refit_by_quarter"].items():
            w = RF["frozen_by_quarter"].get(q, {})
            def q_(x, k, d=3, sign=False):
                z = x.get(k); fmt = (lambda a: f"{a:+.{d}f}") if sign else (lambda a: f"{a:.{d}f}")
                if isinstance(z, dict): return fmt(z["estimate"]) + f" ({fmt(z['ci_95'][0])} to {fmt(z['ci_95'][1])})"
                return fmt(z) if isinstance(z, (int, float)) else ""
            f.write(f"| {q} | {v['n']:,} | {q_(v,'auroc')} | {q_(v,'calibration_intercept',2,True)} | {q_(v,'per_100_flags',1)} | {q_(w,'auroc')} | {q_(w,'calibration_intercept',2,True)} | {q_(w,'per_100_flags',1)} |\n")
    canon["derived"]["refit_summary"] = {"pooled_n": RF["pooled_n"], "refit_auroc": RF["refit_quarterly"]["auroc"], "frozen_auroc": RF["frozen_2025_03_31"]["auroc"], "refit_intercept": RF["refit_quarterly"]["calibration_intercept"], "frozen_intercept": RF["frozen_2025_03_31"]["calibration_intercept"], "refit_per100": RF["refit_quarterly"]["per_100_flags"], "frozen_per100": RF["frozen_2025_03_31"]["per_100_flags"], "refit_flag_rate": RF["refit_quarterly"]["flag_rate"], "frozen_flag_rate": RF["frozen_2025_03_31"]["flag_rate"], "refit_intercept_by_quarter": {q: v["calibration_intercept"] for q, v in RF["refit_by_quarter"].items()}, "frozen_intercept_by_quarter": {q: v["calibration_intercept"] for q, v in RF["frozen_by_quarter"].items()}, "headline_ci": RF["frozen_2025_11_30_with_ci"]}
json.dump(canon, open(R/"canonical.json", "w"), indent=1, default=str); print("intervals, ablation, refit rendered")

# ======================= Blinded physician review =======================
PR = J("physician_review.json")
if PR:
    canon["physician_review"] = PR; canon["generated_from"].append("physician_review.json")
    ex = PR.get("excluding_flagged", {}); pv = PR["prognosis_vs_model"]; ex_pv = ex.get("prognosis_vs_model", {})
    k = {d["pair"]: d["kappa"] for d in PR["agreement"]["q1_engagement_status"]}; kh = {d["pair"]: d["kappa"] for d in PR["agreement"]["q4_harm_if_lost"]}; ki = {d["pair"]: d["kappa"] for d in PR["agreement"]["q5_stated_intent"]}
    canon["derived"]["physician"] = {"forms": PR["phase1_forms"], "cases": PR["cases_reviewed"], "flagged": PR.get("reviewer_quality_flags", {}), "excluded": ex.get("excluded", []), "n_forms_excl": ex.get("n_forms"),
        "kappa_q1": k, "kappa_q4": kh, "kappa_q5": ki, "icc_prognosis_all": PR["icc_prognosis"], "icc_prognosis_excl": ex.get("icc_prognosis"), "fleiss_q1_core": PR.get("fleiss_kappa_q1_core"),
        "auroc_physician_all": pv["auroc_physician"], "auroc_model_cases": pv["auroc_model"], "diff_ci_all": pv["ci_95"], "auroc_physician_excl": ex_pv.get("auroc_physician"), "diff_ci_excl": ex_pv.get("difference_model_minus_physician_ci_95"), "within_stratum_excl": ex.get("prognosis_auroc_within_stratum", {}), "within_stratum_all": PR.get("prognosis_auroc_within_stratum_all", {}), "model_within_stratum": PR.get("model_auroc_within_stratum", {}), "phase2_double": PR.get("phase2_double_coded_agreement"), "phase2_double_kappa": PR.get("phase2_double_coded_kappa"),
        "any_open_need_y1_all": PR["any_open_need_by_outcome_all_reviewers"]["y1"], "any_open_need_y0_all": PR["any_open_need_by_outcome_all_reviewers"]["y0"], "any_open_need_y1_excl": ex.get("any_open_need_by_outcome", {}).get("y1"), "any_open_need_y0_excl": ex.get("any_open_need_by_outcome", {}).get("y0"),
        "harm_high_y1_all": PR["open_need_and_harm_by_outcome"]["y1"]["harm_high"], "harm_high_y0_all": PR["open_need_and_harm_by_outcome"]["y0"]["harm_high"], "harm_high_y1_excl": ex.get("open_need_and_harm_by_outcome", {}).get("y1", {}).get("harm_high"), "harm_high_y0_excl": ex.get("open_need_and_harm_by_outcome", {}).get("y0", {}).get("harm_high"),
        "needs_by_outcome_excl": ex.get("open_need_and_harm_by_outcome", {}), "status_by_outcome_excl": ex.get("engagement_status_by_outcome", {}), "status_by_outcome_all": PR["engagement_status_by_outcome"],
        "phase2_all": PR["phase2_reasons"], "phase2_n_all": PR["phase2_n"], "phase2_excl": ex.get("phase2_reasons", {}), "phase2_n_excl": ex.get("phase2_n"), "phase2_by_reviewer": PR.get("phase2_reasons_by_reviewer", {}),
        "patterns": {r["pattern"]: r["mean"] for r in PR["phase3_pattern_ratings"]}}
    with open(TAB/"tableS15_physician_review.md", "w") as f:
        f.write(f"**Supplementary Table S15. Blinded physician review of {PR['cases_reviewed']} test-era decision points ({PR['phase1_forms']} phase 1 forms from three reviewers; each case reviewed by two, a 30-case core by all three).** Reviewers B and C completed blank forms; reviewer A reviewed and approved forms pre-filled by a keyword-rule procedure (Supplementary Note S7).\n\n")
        f.write("**a. Agreement.**\n\n| Item | Pair | n cases | Kappa |\n|---|---|---|---|\n")
        for item, lab in [("q1_engagement_status","Engagement status"),("q4_harm_if_lost","Harm if lost"),("q5_stated_intent","Stated intent")]:
            for d in PR["agreement"][item]: f.write(f"| {lab} | {d['pair']} | {d['n']} | {d['kappa']:.3f} |\n")
        wo = PR.get("leave_one_reviewer_out", {}).get("without_A", {})
        if wo: f.write(f"| Engagement status, reviewers B and C only (blank forms) | BC | {wo['kappa_q1'][0]['n'] if wo.get('kappa_q1') else ''} | {wo['kappa_q1'][0]['kappa'] if wo.get('kappa_q1') else float('nan'):.3f} |\n")
        f.write(f"| Prognosis (1 to 5), intraclass correlation | all reviewers | | {PR['icc_prognosis']:.3f} |\n| Engagement status, Fleiss kappa on 30-case core | all reviewers | 30 | {PR.get('fleiss_kappa_q1_core', float('nan')):.3f} |\n")
        f.write(f"\n**b. Prognosis against realized disengagement on the same cases.** {PR['design_note']}\n\n| Raters | AUROC of mean physician prognosis | AUROC of the full model on the same cases | Difference (model minus physician), 95% bootstrap CI |\n|---|---|---|---|\n| All three reviewers | {pv['auroc_physician']:.3f} | {pv['auroc_model']:.3f} | {pv['difference_model_minus_physician']:+.3f} ({pv['ci_95'][0]:.3f} to {pv['ci_95'][1]:.3f}) |\n")
        if wo: f.write(f"| Reviewers B and C only | {wo['auroc_physician']:.3f} | {wo['auroc_model']:.3f} | |\n")
        if ex_pv: f.write(f"| Excluding B | {ex_pv['auroc_physician']:.3f} | {ex_pv['auroc_model']:.3f} | {ex_pv['auroc_model']-ex_pv['auroc_physician']:+.3f} ({ex_pv['difference_model_minus_physician_ci_95'][0]:.3f} to {ex_pv['difference_model_minus_physician_ci_95'][1]:.3f}) |\n")
        for st, v in PR.get("prognosis_auroc_within_stratum_all", {}).items(): f.write(f"| All reviewers, within model stratum {st.replace('_',' ')} | {v:.3f} | {PR.get('model_auroc_within_stratum', {}).get(st, float('nan')):.3f} | |\n")
        for st, v in ex.get("prognosis_auroc_within_stratum", {}).items(): f.write(f"| Excluding flagged, within model stratum {st.replace('_',' ')} | {v:.3f} | | |\n")
        f.write("\n**c. Open clinical need and harm if lost, share of cases (mean over reviewers per case), by realized outcome.**\n\n| Item | Disengaged | Retained |\n|---|---|---|\n")
        NM = {"q3_open_need_medication":"Medication gap","q3_open_need_referral_or_appointment":"Pending referral or appointment","q3_open_need_uncontrolled_condition":"Uncontrolled condition","q3_open_need_social_crisis":"Active social crisis","q3_open_need_behavioral_health":"Behavioral health need","q3_no_open_need":"No open need","harm_high":"Harm if lost rated high"}
        for kk, lab in NM.items():
            a1 = PR["open_need_and_harm_by_outcome"]["y1"][kk]; a0 = PR["open_need_and_harm_by_outcome"]["y0"][kk]; e1 = ex.get("open_need_and_harm_by_outcome", {}).get("y1", {}).get(kk); e0 = ex.get("open_need_and_harm_by_outcome", {}).get("y0", {}).get(kk)
            f.write(f"| {lab} | {100*a1:.1f}% | {100*a0:.1f}% |\n")
        f.write(f"| Any open need | {100*PR['any_open_need_by_outcome_all_reviewers']['y1']:.1f}% | {100*PR['any_open_need_by_outcome_all_reviewers']['y0']:.1f}% |\n")
        f.write("\n**d. Engagement status at the index contact by realized outcome (share of forms).**\n\n| Status | Disengaged | Retained |\n|---|---|---|\n")
        for st in ["engaged","ambivalent","declining","unreachable","administrative"]:
            f.write(f"| {st} | {100*PR['engagement_status_by_outcome']['y1'].get(st,0):.1f}% | {100*PR['engagement_status_by_outcome']['y0'].get(st,0):.1f}% |\n")
        f.write(f"\n**e. Adjudicated reason for disengagement (phase 2; one reviewer per case; {PR.get('phase2_double_coded_agreement', {}).get('n_double_coded', 0)} cases double-coded with {100*(PR.get('phase2_double_coded_agreement', {}).get('percent_agreement') or 0):.0f}% agreement).**\n\n| Reason | Share of {PR['phase2_n']} cases |\n|---|---|\n")
        for r in ["lost_contact","administrative_or_coverage","declined_or_dissatisfied","needs_met","cannot_determine","life_event"]:
            f.write(f"| {r.replace('_',' ')} | {100*PR['phase2_reasons'].get(r,0):.1f}% |\n")
        f.write("\n**f. Clinical meaningfulness of the 20 note-text patterns (1 to 5; mean of three reviewers).**\n\n| Pattern | Mean | SD |\n|---|---|---|\n")
        for r in sorted(PR["phase3_pattern_ratings"], key=lambda r: -r["mean"]): f.write(f"| {r['pattern']} | {r['mean']:.2f} | {r['std']:.2f} |\n")
    json.dump(canon, open(R/"canonical.json", "w"), indent=1, default=str); print("physician review rendered")
