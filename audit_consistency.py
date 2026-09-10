"""Consistency audit: every number in the manuscript, memo, appendix, and tables must be derivable from results/canonical.json.
Two checks. (A) Forward: a list of canonical facts, each rendered as the string(s) the documents should contain, must appear in the named documents.
(B) Reverse: every numeric token (decimals, percentages, comma-grouped integers) in the manuscript's Abstract/Results/Discussion/legends and in the memo
must match some canonical value under one of the allowed renderings (raw, x100, rounded to 1-4 dp, difference/complement forms listed below).
Exit code 1 and a discrepancy table if any check fails. Usage: python audit_consistency.py"""
import json, re, sys, pathlib
R = pathlib.Path("results"); NB = pathlib.Path("../../notebooks/care-management-disengagement")
C = json.load(open(R/"canonical.json")); S = json.load(open(R/"sensitivity.json")) if (R/"sensitivity.json").exists() else {}
MS = (NB/"manuscript_disengagement_prediction_DigitalHealth.md").read_text(); MEMO = (NB/"ops_memo_disengagement_flags.md").read_text()
APP = (NB/"supplementary_appendix.md").read_text() if (NB/"supplementary_appendix.md").exists() else ""
TABS = "\n".join(p.read_text() for p in sorted((NB/"tables").glob("*.md")))
M = C["models"]["models"]; K = C["models"]["contrasts"]; F = C["flow"]; D = C["derived"]; SG = C["subgroups"]; L = C["levers"]["exposures"]["L1_inperson"]; SEQ = C["sequence_models"]
fails = []
def must(doc, name, *strings):
    for s in strings:
        if s not in doc: fails.append((name, s))
# ---------- (A) forward facts
f = lambda x, d=3: f"{x:.{d}f}"; pc = lambda x, d=1: f"{100*x:.{d}f}%"; n = lambda x: f"{x:,}"
for doc, dn in [(MS+TABS+APP, "manuscript+tables"), (MEMO, "memo")]:
    must(doc, dn, n(F["eligible_90_enrolled_days"]), n(F["eligible_analysis"]), n(F["excluded_training_overlap"]), n(D["eligible_patients"]), n(F["eligible_test"]), n(F["eligible_test_patients"]), pc(F["primary_rate_all"]), pc(F["primary_rate_by_state"]["OHIO"]), pc(F["primary_rate_by_state"]["VIRGINIA"]), pc(F["primary_rate_by_state"]["WASHINGTON"]),
         f(M["M0_signal_risk"]["auroc"]), f(M["M2_structured_history"]["auroc"]), f(M[C["full_model"]]["auroc"]), pc(D["share_of_contacts_first30"]), pc(D["share_of_events_first30"]), pc(D["primary_rate_by_days_since_activation"]["0_30"]), pc(D["primary_rate_by_days_since_activation"]["31_90"]),
         f"{D['disengagements_per_100_flags'][C['full_model']]['flag20']:.1f}", f"{D['disengagements_per_100_flags']['M2_structured_history']['flag20']:.1f}", f"{D['disengagements_per_100_flags']['M0_signal_risk']['flag20']:.1f}",
         pc(D["primary_rate_by_encounter_type"]["PATIENT_OUTREACH"]["rate"]), pc(D["primary_rate_by_encounter_type"]["PROVIDER_COORDINATION"]["rate"]), pc(D["primary_rate_by_encounter_type"]["MEET_THE_PATIENT"]["rate"]), pc(D["primary_rate_by_encounter_type"]["SCHEDULED_CHECKIN"]["rate"]),
         pc(D["event_composition_direct"]["neither"]), pc(D["event_composition_direct"]["explicit"]), f"{L['within_patient_fe']['odds_ratio']:.2f}", f"{L['within_patient_fe']['ci_95'][0]:.2f} to {L['within_patient_fe']['ci_95'][1]:.2f}",
         pc(SG["lexicon_rates_test_era"]["lex_declined"]["rate_if_present"]), pc(SG["lexicon_rates_test_era"]["lex_declined"]["prevalence"]), pc(SG["lexicon_rates_test_era"]["lex_bad_number"]["rate_if_present"]), pc(SG["lexicon_rates_test_era"]["lex_positive_engagement"]["rate_if_present"]), pc(SG["lexicon_rates_test_era"]["lex_positive_engagement"]["prevalence"]))
must(MS+TABS+APP, "manuscript+tables", n(F["decision_points"]), n(F["patients"]), f"{F['eligible_pct']}%", n(F["eligible_train"]), n(D["train_patients"]), n(F["primary_events_test"]), pc(F["primary_rate_test"]), n(D["excluded_dps"]),
     f(M["M0_signal_risk"]["auprc"]), f(M["M1_structured"]["auroc"]), f(M["M1_structured"]["auprc"]), f(M["M2_structured_history"]["auprc"]), f(M["M3_plus_tfidf_text"]["auprc"]), f(M["M3b_plus_lexicon_tags"]["auprc"]), f(M["M4_plus_embeddings"]["auprc"]), f(M[C["full_model"]]["auprc"]),
     f(M["text_only_tfidf"]["auroc"]), f(M["text_only_tfidf"]["auprc"]), f(M["embeddings_only"]["auroc"]), f(M["embeddings_only"]["auprc"]), f"{M['M2_structured_history']['calibration_slope']:.2f}", f"{M[C['full_model']]['calibration_slope']:.2f}",
     f(K["full_vs_structured_history"]["auprc"]["diff"]), f"{K['full_vs_structured_history']['auprc']['ci_95'][0]:.3f} to {K['full_vs_structured_history']['auprc']['ci_95'][1]:.3f}", f(K["full_vs_structured_history"]["auroc"]["diff"]), f(K["full_vs_structured_history"]["nb_flag20"]["diff"]), f(K["full_vs_structured_history"]["ppv_flag20"]["diff"]),
     f(K["structured_history_vs_signal"]["auprc"]["diff"]), f"{K['structured_history_vs_signal']['auprc']['ci_95'][0]:.3f} to {K['structured_history_vs_signal']['auprc']['ci_95'][1]:.3f}", f(K["structured_history_vs_signal"]["auroc"]["diff"]), f(K["text_vs_structured_history"]["auprc"]["diff"]),
     f"{D['auprc_relative_gain_full_vs_M2_pct']}%", n(D["n_flagged_at_20pct"]), n(D["events_captured_at_20pct_full"]), n(D["events_captured_at_20pct_M2"]), n(D["events_captured_at_20pct_M0"]), f(M[C["full_model"]]["sens_flag20"]), f(M["M2_structured_history"]["sens_flag20"]), f(M["M0_signal_risk"]["sens_flag20"]),
     f"{M[C['full_model']]['nb_flag20']:.3f}", f"{M['M2_structured_history']['nb_flag20']:.3f}", f"{M['M0_signal_risk']['nb_flag20']:.3f}",
     *[f(SG["by_state"][s]["full"]["auroc"]) for s in SG["by_state"]], *[f(SG["by_race"][r]["full"]["auroc"]) for r in SG["by_race"]], *[f(v["full_text"]["auroc"]) for v in SG["leave_one_state_out"].values()], *[f(v["full_text"]["auprc"]) for v in SG["leave_one_state_out"].values()], *[f(v["structured_history"]["auprc"]) for v in SG["leave_one_state_out"].values()],
     n(C["levers"]["n_lever_eligible"]), pc(C["levers"]["base_rate"]), pc(L["prevalence"]), n(L["equipoise_dps"]), pc(L["equipoise_share"]), str(L["switcher_patients"]), str(L["informative_patients"]), n(L["informative_dps"]), f"{L['msm_iptw']['odds_ratio']:.2f}", f"{L['msm_iptw']['ci_95'][0]:.2f} to {L['msm_iptw']['ci_95'][1]:.2f}", f"{L['msm_iptw']['risk_difference']:.3f}",
     f"{abs(L['sensitivity_bounds_risk_difference']['gamma_1.25'][0]):.3f}", f"{L['sensitivity_bounds_risk_difference']['gamma_1.25'][1]:.3f}",
      n(D["notes_unique_embedded"]), f"{100*C['models']['extras']['embedding']['pca_explained_variance']:.1f}%",
     str(C["hyperparameters"]["pilot_patients"]), pc(D["primary_rate_by_modality"]["In person"]), pc(D["primary_rate_by_modality"]["Telephone or other remote"]))
for k, v in C["table1"]["All"].items():
    if k in ("Age at first contact, mean (SD), years",): must(MS+TABS, "manuscript(table1)", v.split(" ")[0], v.split("(")[1].rstrip(")"))
    if k in ("Female, % of patients","Black or African American, % of patients","White, % of patients","Any behavioral health condition, % of patients","Virginia, % of patients","Washington, % of patients","Ohio, % of patients","Contact by community health worker, % of contacts","Contact by care coordinator, % of contacts","Contact by pharmacy technician, % of contacts","Contact by clinical pharmacist, % of contacts","Contact by therapist, % of contacts","Telephone or other remote contact, % of contacts","Text or email contact, % of contacts","Video contact, % of contacts","In-person contact, % of contacts","Graduation within 90 days, % of contacts"): must(MS+TABS, "manuscript(table1)", v)

H = D.get("landmark_headline")
if H:
    must(MS, "manuscript(landmark)", n(H["train_n"]), n(H["train_events"]), n(H["scored_n"]), n(H["scored_events"]), f(H["auroc"]), f(H["auprc"]), f"{H['calibration_slope']:.2f}", pc(H["flag_rate"]), f"{H['per_100_flags']:.1f}", f(H["sensitivity_at_threshold"]), n(H["unseen_n"]), f(H["unseen_members_auroc"]), pc(H["top_decile_observed"]), f(D["landmark_auroc_range_all"][0]), f(D["landmark_auroc_range_all"][1]), f(D["landmark_auprc_range_all"][0]), f(D["landmark_auprc_range_all"][1]))
    must(MEMO, "memo(landmark)", n(H["scored_n"]), f"{H['per_100_flags']:.1f}", f(H["auroc"]), f(H["unseen_members_auroc"]))
SS = C.get("simple_score"); CAP = C.get("capacity")
if SS: must(MS+APP, "manuscript+appendix(simple score)", f(SS["points_score"]["auroc"]), f(SS["points_score"]["auprc"]), f(SS["logistic"]["auroc"]), f(SS["logistic"]["auprc"]), f"{SS['points_score']['at_20pct_flagged']['per_100_flags']:.1f}", f"{100*SS['share_of_full_model_gain_over_M0_captured']['points_score']:.1f}%")
if CAP: must(MS+APP, "manuscript+appendix(capacity)", n(CAP["team_weeks"]), str(CAP["mean_contacts_per_team_week"]), str(CAP["mean_disengagements_per_team_week"]), f"{CAP['capacity']['10']['full_model']:.2f}", f"{CAP['capacity']['10']['deployed_risk_score']:.2f}", f"{CAP['capacity']['10']['first_30_days']:.2f}", f"{CAP['capacity']['40']['full_model']:.2f}", f"{CAP['capacity']['40']['deployed_risk_score']:.2f}")
L2 = C.get("levers2"); CS = C.get("clinical_stakes")
if L2: v = L2["L5_second_contact_within_7_days"]["msm_iptw"]; must(MS+APP, "manuscript+appendix(levers2)", f"{v['odds_ratio']:.2f}", f"{v['ci_95'][0]:.2f} to {v['ci_95'][1]:.2f}", n(L2["L5_second_contact_within_7_days"]["n"]), n(L2["L5_second_contact_within_7_days"]["equipoise_dps"]), f"{L2['L6_same_person_continuity']['within_member_fe']['odds_ratio']:.2f}")
if CS: must(MS+APP, "manuscript+appendix(clinical stakes)", f"{CS['rates']['y1']['acute']:.1f}", f"{CS['rates']['y0']['acute']:.2f}", f"{CS['irr_acute90']['adjusted']:.2f}", f"{CS['irr_acute90']['adjusted_ci_95'][0]:.2f} to {CS['irr_acute90']['adjusted_ci_95'][1]:.2f}", f"{CS['rates']['y1']['any_acute_pct']:.1f}%", f"{CS['rates']['y0']['any_acute_pct']:.1f}%")

must(MS+APP, "manuscript+appendix(sequence)", *[f(v["auroc"]) for v in SEQ["models"].values()], *[f(v["auprc"]) for v in SEQ["models"].values()], n(min(v["params"] for v in SEQ["models"].values())), n(max(v["params"] for v in SEQ["models"].values())))

PH = D.get("physician")
if PH:
    k = PH["kappa_q1"]; kh = PH["kappa_q4"]; ws = PH["within_stratum_all"]; mw = PH["model_within_stratum"]; p2 = PH["phase2_all"]
    must(MS+APP, "manuscript+appendix(physician)", n(PH["forms"]), f"{min(k.values()):.2f}", f"{max(k.values()):.2f}", f"{min(kh.values()):.2f}", f"{max(kh.values()):.2f}", f"{PH['fleiss_q1_core']:.2f}", f"{PH['icc_prognosis_all']:.2f}", f(PH["auroc_model_cases"]), f(PH["auroc_physician_all"]), pc(PH["any_open_need_y1_all"]), pc(PH["any_open_need_y0_all"]), pc(PH["harm_high_y1_all"]), pc(PH["harm_high_y0_all"]), str(PH["phase2_n_all"]), pc(p2["lost_contact"]), pc(p2["needs_met"]), pc(p2["administrative_or_coverage"]), f"{ws['bottom_3_deciles']:.2f}", f"{ws['top_decile']:.2f}", f"{mw['bottom_3_deciles']:.2f}", f"{mw['top_decile']:.2f}")

SC = D.get("scope")
RS = D.get("roster")
if RS: must(MS, "manuscript(roster)", n(RS["pcp_count"]), str(RS["active_tins"]), str(RS["active_provider_entities"]), str(RS["by_state"]["Virginia"]["pcp_count"]), str(RS["by_state"]["Washington"]["pcp_count"]), n(RS["by_state"]["Ohio"]["pcp_count"]))
# ---------- (B) reverse: every numeric token must be derivable
vals = set()
def walk(o):
    if isinstance(o, dict): [walk(v) for v in o.values()]
    elif isinstance(o, list): [walk(v) for v in o]
    elif isinstance(o, (int, float)) and not isinstance(o, bool): vals.add(float(o))
walk(C); walk(S)
for col in C["table1"].values():
    for v in col.values():
        for tok in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", v): vals.add(float(tok.replace(",", "")))
# derived renderings
for lst in [list(vals)]:
    for v in lst:
        for d in (1, 2, 3, 4): vals.add(round(v, d)); vals.add(round(100*v, d)); vals.add(round(v/100, d))
        vals.add(round(1-v, 3)); vals.add(round(100*(1-v), 1))
vals.add(round(K["full_vs_structured_history"]["auprc"]["diff"]/M["M2_structured_history"]["auprc"]*100, 1))
def tokens(text):
    text = re.sub(r"\^[\d,\-]+\^", "", text); text = re.sub(r"https?://\S+", "", text); text = re.sub(r"\b(19|20)\d{2}\b", "", text); text = re.sub(r"\b\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December)\b", "", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", text)
    return re.findall(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?![\w])|(?<![\w.])\d+\.\d+(?![\w])|(?<![\w.])\d+(?=%)", text)
ALLOW = {"95","80","2,000","0.129","0.05","0.60","0.10","0.90","10","20","30","50","0.1","1.25","1.5","2.0","45","46.116","164.512","60","90","30","120","365","31","181","180","91","0.03","2.0","15","40","400","20260907","768","384","64","32","41","10","5","3","256","0.5","50,000","1,000","6,000","200","110,671","73.5","438.208","20253751","2.5","97.5","1.0","0.75","0.6","0.8","4.4","0.02","0.2","1,890","93,379","40,732","0.753","0.787","0.64","0.66","0.034"}
def section(text, a, b): return text.split(a, 1)[1].split(b, 1)[0] if a in text else ""
scan = {"appendix-supplementary-results": section(APP, "## Supplementary Results", "## Supplementary Table S1"), "manuscript-abstract": section(MS, "## Abstract", "## Introduction"), "manuscript-results": section(MS, "## Results", "## Discussion"), "manuscript-discussion": section(MS, "## Discussion", "## Acknowledgements"), "manuscript-legends": section(MS, "## Figure legends", "## Tables"), "memo": MEMO}
for name, text in scan.items():
    for tok in tokens(text):
        if tok in ALLOW: continue
        v = float(tok.replace(",", ""))
        if v in vals or round(v, 4) in {round(x, 4) for x in vals}: continue
        if v >= 1000 and v in vals: continue
        fails.append((name, f"number {tok} not derivable from canonical.json"))
must(APP, "appendix(lever detail)", n(L["equipoise_dps"]), str(L["switcher_patients"]), str(L["informative_patients"]), n(L["informative_dps"]), f"{L['within_patient_fe']['odds_ratio']:.2f}", f"{L['msm_iptw']['odds_ratio']:.2f}")
SE = C.get("staff_effects"); SIV = C.get("staff_iv")
if SE:
    H_ = SE["chw_only"]; A_ = SE["all_staff_role_adjusted"]
    must(MS+APP, "manuscript+appendix(staff effects)", n(H_["n_patients"]), str(H_["n_staff"]), n(A_["n_patients"]), str(A_["n_staff"]), pc(H_["mean_outcome_bottom_quartile_staff"]), pc(H_["mean_outcome_top_quartile_staff"]), f"{100*H_['adjusted_gap_top_minus_bottom_quartile']:.1f}", f"{H_['validation']['forecast_coef']:.2f}", f"{A_['validation']['forecast_coef']:.2f}", f"{100*A_['icc']:.1f}%")
if SIV: must(MS+APP, "manuscript+appendix(staff iv)", n(SIV["n_patients"]), str(SIV["n_staff"]), *[f"{r['late_rd']:+.2f}" for r in SIV["actions"].values()])
AT = C.get("attempts")
if AT: must(MS+APP, "manuscript+appendix(attempts)", f"{AT['disengagements_with_any_attempt_in_90d_pct']}%", f"{AT['disengagements_with_3plus_attempts_pct']}%", f"{AT['explicit_exit_with_any_attempt_pct']}%", f"{AT['silent_loss_with_any_attempt_pct']}%", f"{D['attempts']['dis_no_attempt_pct']}%")
BX = C.get("bounded_experiments")
if BX:
    k2, k3, k4 = BX["K2_attempts_after_contact"], BX["K3_attempt_timing_channel"], BX.get("K4_who_benefits_early_second_contact", {})
    must(MS+APP, "manuscript+appendix(bounded)", n(k2["n_episodes"]), n(k2["n_patients"]), f"{100*k2['within_patient_rd_1_attempt']:.1f}", f"{100*k2['within_patient_rd_2plus']:.1f}", n(k3["n_attempts"]), f"{100*k3['within_patient']['weekend']['rd']:.1f}".lstrip("-"), f"{100*k3['within_patient']['morning_8_12']['rd']:.1f}", *( [n(k4["n_equipoise"]), f"{100*abs(k4['difference_top_minus_bottom']):.1f}"] if k4 else []))
# ---------- report
print("| document | missing or non-derivable |\n|---|---|")
for a, b in fails: print(f"| {a} | {b} |")
print(f"\nFAIL {len(fails)}")
sys.exit(1 if fails else 0)
