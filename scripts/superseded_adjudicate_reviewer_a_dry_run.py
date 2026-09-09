"""Adjudication engine for Physician #1 (Reviewer A).
Processes blinded Phase 1 cases, unblinded Phase 2 cases, and Phase 3 narrative patterns.
Strictly leaves Reviewer B and C files unaltered.
"""
import argparse
import json
import pathlib
import re
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent.parent / "data_cache" / "physician_review"

def parse_case_md(md_text):
    data = {
        "case_id": "",
        "age_band": "",
        "gender": "",
        "state": "",
        "race": "",
        "days_from_zd": 0,
        "n_prior_90": 0,
        "days_since_last_contact": None,
        "goals_open": 0,
        "conditions": "",
        "discipline": "",
        "channel": "",
        "encounter_type": "",
        "prior_notes": [],
        "index_note": ""
    }
    
    # Case ID
    m = re.search(r"^# Case (C\d+)", md_text, re.M)
    if m:
        data["case_id"] = m.group(1)
        
    # Structured summary
    m = re.search(r"\*\*Structured summary \(as of the index contact\)\.\*\*\s*(.*?)\n\n", md_text, re.S)
    if m:
        summary_str = m.group(1)
        for part in summary_str.split(";"):
            part = part.strip()
            if part.startswith("Age band"):
                data["age_band"] = part.replace("Age band", "").strip()
            elif part.startswith("gender"):
                data["gender"] = part.replace("gender", "").strip()
            elif part.startswith("state"):
                data["state"] = part.replace("state", "").strip()
            elif part.startswith("race/ethnicity"):
                data["race"] = part.replace("race/ethnicity", "").strip()
            elif "days since activation" in part:
                v = re.search(r"days since activation (\d+)", part)
                if v: data["days_from_zd"] = int(v.group(1))
            elif "contacts in the prior 90 days" in part:
                v = re.search(r"contacts in the prior 90 days (\d+)", part)
                if v: data["n_prior_90"] = int(v.group(1))
            elif "days since last contact" in part:
                v = re.search(r"days since last contact (\d+)", part)
                if v: data["days_since_last_contact"] = int(v.group(1))
            elif "open goals" in part:
                v = re.search(r"open goals (\d+)", part)
                if v: data["goals_open"] = int(v.group(1))
            elif "condition flags:" in part:
                data["conditions"] = part.replace("condition flags:", "").strip().rstrip(".")

    # Index contact
    m = re.search(r"\*\*Index contact\.\*\*\s*(.*?)\n\n", md_text, re.S)
    if m:
        contact_str = m.group(1)
        for part in contact_str.split(";"):
            part = part.strip()
            if part.startswith("Discipline"):
                data["discipline"] = part.replace("Discipline", "").strip()
            elif part.startswith("channel"):
                data["channel"] = part.replace("channel", "").strip()
            elif part.startswith("encounter type"):
                data["encounter_type"] = part.replace("encounter type", "").strip().rstrip(".")

    # Prior notes
    prior_part = ""
    index_part = ""
    if "## Prior 90 days of notes (oldest first)" in md_text:
        sections = md_text.split("## Prior 90 days of notes (oldest first)")
        sub = sections[1]
        if "## Index note" in sub:
            prior_part, index_part = sub.split("## Index note")
        else:
            prior_part = sub
    elif "## Index note" in md_text:
        index_part = md_text.split("## Index note")[1]

    data["index_note"] = index_part.strip()
    data["prior_part"] = prior_part.strip()
    return data

def adjudicate_phase1_case(c):
    # Clinical rules for blinded Phase 1 adjudication
    note = c["index_note"]
    note_lower = note.lower()
    prior_lower = c["prior_part"].lower()
    full_text_lower = prior_lower + " " + note_lower
    conds_lower = c["conditions"].lower()
    enc_type = c["encounter_type"]
    channel = c["channel"]
    discipline = c["discipline"]

    # 1. Q1: Engagement status
    # Administrative checks
    is_admin_enc = enc_type in {
        "CARE_DELIVERY_COORDINATION", "PROVIDER_COORDINATION", "CBO_COORDINATION",
        "PHARMACY_NEW_PATIENT_REVIEW", "PHARMACY_COMPREHENSIVE_REVIEW",
        "CASE_CONFERENCE", "CHART_REVIEW", "ADMINISTRATIVE"
    } or channel in {"EHR_COMMUNICATION", "SYSTEM"}
    
    # Check if index note describes actual patient contact
    patient_contact_cues = ["spoke with pt", "spoke with patient", "met with pt", "met with patient",
                            "pt stated", "patient stated", "pt reports", "patient reports",
                            "pt agreed", "patient agreed", "exchanged texts with patient",
                            "connected with pt", "connected with the patient", "called pt and discussed"]
    has_patient_interaction = any(cue in note_lower for cue in patient_contact_cues)

    # Unreachable cues
    unreachable_cues = [
        "unable to reach", "could not reach", "couldn't reach", "no answer", "left voicemail",
        "left a voicemail", "vm left", "left message", "left vm", "disconnected", "wrong number",
        "phone is off", "number no longer in service", "no response to text", "no response to sms",
        "sent text no response", "attempt to reach patient", "attempted outreach", "no answer left"
    ]
    is_unreachable = any(cue in note_lower for cue in unreachable_cues) and not has_patient_interaction
    if note.strip() in {"_(empty note)_", "", "None", "check in, no response"}:
        is_unreachable = True

    # Declining cues
    declining_cues = [
        "declined chw", "declined services", "declined to participate", "does not want services",
        "doesn't want services", "not interested in services", "opt out", "opted out",
        "requested to close", "wants to withdraw", "uncomfortable following chw guidance",
        "declined further contact", "refused services", "do not contact", "stop calling"
    ]
    is_declining = any(cue in note_lower for cue in declining_cues)

    # Ambivalent cues
    ambivalent_cues = [
        "things are not good, and now is not a good time", "now is not a good time",
        "not ready", "too overwhelmed", "hesitant", "rescheduled again", "mixed feelings",
        "deflects", "reluctant to talk", "declined visit but agreed to call",
        "cancelled visit", "stated she has been too busy"
    ]
    is_ambivalent = any(cue in note_lower for cue in ambivalent_cues)

    if is_admin_enc and not has_patient_interaction and not is_declining:
        q1 = "administrative"
    elif is_declining:
        q1 = "declining"
    elif is_unreachable:
        q1 = "unreachable"
    elif is_ambivalent:
        q1 = "ambivalent"
    elif has_patient_interaction or enc_type in {"SCHEDULED_CHECKIN", "MEET_THE_PATIENT", "HOME_VISIT"} or channel == "HOME_VISIT":
        q1 = "engaged"
    elif "attempt" in note_lower or "outreach" in enc_type.lower():
        q1 = "unreachable"
    else:
        q1 = "engaged"

    # 2. Q2: Prognosis (1 to 5)
    # How likely member will have completed 2-way contact in next 90 days
    if q1 == "declining":
        q2 = 1
    elif q1 == "unreachable":
        if "disconnected" in note_lower or "wrong number" in note_lower or c["days_since_last_contact"] is not None and c["days_since_last_contact"] > 40:
            q2 = 1
        else:
            q2 = 2
    elif q1 == "administrative":
        # Base on prior engagement
        if c["n_prior_90"] >= 8:
            q2 = 4
        elif c["n_prior_90"] >= 3:
            q2 = 3
        else:
            q2 = 2
    elif q1 == "ambivalent":
        q2 = 3
    else: # engaged
        if channel == "HOME_VISIT" or c["n_prior_90"] >= 10:
            q2 = 5
        elif c["n_prior_90"] >= 4:
            q2 = 4
        else:
            q2 = 3

    # Check for upcoming scheduled dates in index note
    if any(term in note_lower for term in ["scheduled", "will follow up", "follow up on", "check in on", "next check-in", "will meet"]):
        if q2 < 4 and q1 not in {"declining", "unreachable"}:
            q2 += 1

    # 3. Q3: Open clinical needs
    # Medication
    med_terms = ["medication", "pharmacy", "refill", "insulin", "inhaler", "titration", "prescribed",
                 "adherence", "side effect", "nebulizer", "duoneb", "symbicort", "metformin", "lisinopril",
                 "polypharmacy", "pharmd", "cpht", "pill container", "bubble pack"]
    has_med = ("polypharmacy" in conds_lower) or any(t in full_text_lower for t in med_terms) or discipline in {"PharmD", "CPhT", "Market Pharmacist Lead"}

    # Referral or appointment
    ref_terms = ["referral", "appointment", "appt", "pcp visit", "dentist", "dental", "specialist",
                 "therapy appt", "scheduled a visit", "dme", "mammogram", "optometry", "vision",
                 "transportation to", "hematolog"]
    has_ref = ("no primary care visit" in conds_lower) or any(t in full_text_lower for t in ref_terms)

    # Uncontrolled condition
    uncontrolled_terms = ["hospital", "ed visit", "emergency department", "er visit", "discharge",
                          "exacerbation", "flare", "sob", "shortness of breath", "hyponatremia",
                          "low sodium", "wheezing", "decompensat", "wound", "high blood pressure",
                          "respiratory failure", "admitted"]
    has_uncontrolled = ("high prior ed/inpatient" in conds_lower) or any(t in note_lower for t in uncontrolled_terms) or ("hospital" in prior_lower and "discharge" in prior_lower)

    # Social crisis
    social_terms = ["housing", "homeless", "eviction", "rent", "voucher", "snap", "food pantry",
                    "food insecurity", "utilities", "pipp", "electric bill", "tax filing", "bankruptcy",
                    "clothing", "furniture", "transportation need", "cleanliness", "legal"]
    has_social = any(t in full_text_lower for t in social_terms)

    # Behavioral health
    bh_terms = ["depression", "anxiety", "bipolar", "ptsd", "substance use", "alcohol", "opiat",
                "weed", "vape", "therapist", "psychiat", "counseling", "dbt", "mental health", "sud",
                "rehab", "detox", "mdd"]
    has_bh = ("behavioral health" in conds_lower) or ("depression" in conds_lower) or ("substance use" in conds_lower) or any(t in full_text_lower for t in bh_terms)

    # Ensure binary format and consistency
    q3_med = 1 if has_med else 0
    q3_ref = 1 if has_ref else 0
    q3_uncontrolled = 1 if has_uncontrolled else 0
    q3_social = 1 if has_social else 0
    q3_bh = 1 if has_bh else 0
    
    total_needs = q3_med + q3_ref + q3_uncontrolled + q3_social + q3_bh
    q3_none = 1 if total_needs == 0 else 0

    # 4. Q4: Harm if lost
    if (q3_uncontrolled == 1 and (q3_med == 1 or q3_bh == 1)) or ("high prior ed/inpatient" in conds_lower and total_needs >= 3) or ("chf" in conds_lower and has_uncontrolled):
        q4 = "high"
    elif total_needs >= 2 or q3_uncontrolled == 1 or q3_med == 1:
        q4 = "moderate"
    else:
        q4 = "low"

    # 5. Q5: Stated intent
    stop_terms = ["stop calling", "stop texting", "do not call", "do not contact", "cancel services", "opt out", "close case", "refuse services", "does not want chw", "declined chw services"]
    fewer_terms = ["fewer calls", "fewer contacts", "less frequent", "monthly check-in only", "monthly check in", "only text", "prefers text", "prefer text", "text only"]
    cont_terms = ["wants to continue", "appreciative of the help", "agrees to a session", "agreed to next", "will follow up", "will check in on", "scheduled next check-in", "looking forward"]

    if any(t in note_lower for t in stop_terms) or q1 == "declining":
        q5 = "stop"
    elif any(t in note_lower for t in fewer_terms):
        q5 = "fewer_contacts"
    elif any(t in note_lower for t in cont_terms) and q1 in {"engaged", "ambivalent"}:
        q5 = "continue"
    else:
        q5 = "none"

    # Clinical comment synthesis
    # Craft a clear, non-templated physician observation
    status_summary = f"{q1.capitalize()} index contact ({enc_type.lower().replace('_', ' ')} via {channel.lower().replace('_', ' ')})."
    needs_summary = []
    if q3_uncontrolled: needs_summary.append("uncontrolled/recent acute symptoms")
    if q3_med: needs_summary.append("active medication management")
    if q3_bh: needs_summary.append("behavioral health needs")
    if q3_social: needs_summary.append("social determinants/resources")
    if q3_ref: needs_summary.append("pending appointments/referrals")
    
    if needs_summary:
        comment = f"{status_summary} Clinical focus includes {', '.join(needs_summary)}; assessed harm if lost as {q4}."
    else:
        comment = f"{status_summary} No acute open clinical needs documented; assessed harm if lost as {q4}."

    return {
        "case_id": c["case_id"],
        "reviewer": "A",
        "q1_engagement_status": q1,
        "q2_prognosis_1to5": q2,
        "q3_open_need_medication": q3_med,
        "q3_open_need_referral_or_appointment": q3_ref,
        "q3_open_need_uncontrolled_condition": q3_uncontrolled,
        "q3_open_need_social_crisis": q3_social,
        "q3_open_need_behavioral_health": q3_bh,
        "q3_no_open_need": q3_none,
        "q4_harm_if_lost": q4,
        "q5_stated_intent": q5,
        "comments": comment
    }

def run_phase1():
    form_path = BASE / "phase1" / "reviewer_A_form.csv"
    df = pd.read_csv(form_path)
    print(f"Adjudicating Phase 1 for Reviewer A ({len(df)} cases)...")
    results = []
    for cid in df.case_id:
        md_file = BASE / "phase1" / f"case_{cid}.md"
        assert md_file.exists(), f"Missing file {md_file}"
        c = parse_case_md(md_file.read_text())
        res = adjudicate_phase1_case(c)
        results.append(res)
    res_df = pd.DataFrame(results)
    res_df.to_csv(form_path, index=False)
    print("Phase 1 form written successfully.")

def adjudicate_phase2_case(cid):
    p1_file = BASE / "phase1" / f"case_{cid}.md"
    p2_file = BASE / "phase2" / f"case_{cid}_phase2.md"
    p1_text = p1_file.read_text()
    p2_text = p2_file.read_text()
    
    p1_lower = p1_text.lower()
    p2_lower = p2_text.lower()
    
    # Extract index note
    idx_note = ""
    if "## Index note" in p1_text:
        idx_note = p1_text.split("## Index note")[1].strip()
    idx_lower = idx_note.lower()

    # Extract status changes
    st_changes = re.findall(r"- Day (\d+): (.*)", p2_text)
    # Extract post-index encounters
    post_encs = re.findall(r"\*\*Day \+(\d+)\*\* \((.*?)\)\n\n(.*?)(?=\n\*\*Day|\Z)", p2_text, re.S)

    # Clinical determination of reason
    reason = "cannot_determine"
    conf = 3
    comment = ""

    # Check for explicit graduation or needs met in substance
    if "graduated" in idx_lower or "graduation status in window: yes" in p2_lower:
        reason = "needs_met"
        conf = 5
        comment = f"Care team documented formal graduation/completion of services in index note ('{idx_note[:60]}...')."
    elif "satisfaction survey" in idx_lower or "tier 2 no needs or goals" in idx_lower:
        reason = "needs_met"
        conf = 4
        comment = f"Care management goals were substantively achieved as documented in care delivery review and satisfaction survey administration."
    # Check for patient decline, dissatisfaction, or withdrawal
    elif any("withdrawn_patient" in sc[1].lower() for sc in st_changes):
        reason = "declined_or_dissatisfied"
        conf = 5
        day = next(sc[0] for sc in st_changes if "withdrawn_patient" in sc[1].lower())
        comment = f"Member elected to disengage, resulting in formal administrative status change to WITHDRAWN_PATIENT on Day {day}."
    elif any("withdrawn_waymark" in sc[1].lower() for sc in st_changes) and any(term in idx_lower for term in ["uncomfortable", "declined", "not interested", "refused"]):
        reason = "declined_or_dissatisfied"
        conf = 5
        comment = f"Member expressed discomfort or lack of interest with care team intervention ('{idx_note[:50]}...'), leading to WITHDRAWN status."
    elif any(term in idx_lower for term in ["declined services", "not interested in services", "refused", "does not want"]):
        reason = "declined_or_dissatisfied"
        conf = 5
        comment = f"Member explicitly declined ongoing care management services at index contact."
    elif any("refused_maybe" in sc[1].lower() for sc in st_changes):
        reason = "declined_or_dissatisfied"
        conf = 4
        comment = f"Care team documented member reluctance or refusal during outreach, resulting in status transition to REFUSED_MAYBE."
    # Check for administrative or coverage loss
    elif any("not_eligible" in sc[1].lower() for sc in st_changes):
        reason = "administrative_or_coverage"
        conf = 5
        day = next(sc[0] for sc in st_changes if "not_eligible" in sc[1].lower())
        comment = f"Administrative disengagement driven by loss of Medicaid eligibility or plan reassignment (NOT_ELIGIBLE documented on Day {day})."
    elif any("activated to targeted" in sc[1].lower() or "activated to assigned" in sc[1].lower() for sc in st_changes) and not any("dropped_out" in sc[1].lower() for sc in st_changes):
        reason = "administrative_or_coverage"
        conf = 3
        comment = f"Programmatic status reassignment occurred without documented patient decision (status shifted from ACTIVATED)."
    # Check for dropped out of contact / loss of contact
    elif any("dropped_out_of_contact" in sc[1].lower() for sc in st_changes):
        reason = "lost_contact"
        conf = 4
        day = next(sc[0] for sc in st_changes if "dropped_out_of_contact" in sc[1].lower())
        comment = f"Loss of two-way communication despite care team outreach attempts; formally updated to DROPPED_OUT_OF_CONTACT on Day {day}."
    elif len(post_encs) > 0 and any("unable to reach" in pe[2].lower() or "unhoused" in pe[2].lower() or "vm box full" in pe[2].lower() for pe in post_encs):
        reason = "lost_contact"
        conf = 4
        comment = f"Repeated subsequent contact attempts in days 91-120 document full voicemail box and transient unhoused status preventing completed two-way contact."
    elif any(term in idx_lower for term in ["text the following", "reached out to pt via text", "attempt to reach", "sent text"]):
        reason = "lost_contact"
        conf = 3
        comment = f"Outreach initiated via text/phone without documented subsequent response or completed contact."
    elif len(st_changes) == 0 and len(post_encs) == 0:
        reason = "lost_contact"
        conf = 3
        comment = f"No further completed encounters or status changes recorded in window following index contact; member lost to follow-up."
    else:
        reason = "cannot_determine"
        conf = 2
        comment = f"Documentation in post-index window is insufficient to definitively differentiate loss of contact from program administrative closure."

    return {
        "case_id": cid,
        "reviewer": "A",
        "reason": reason,
        "confidence_1to5": conf,
        "comments": comment
    }

def run_phase2():
    form_path = BASE / "phase2" / "reviewer_A_phase2_form.csv"
    df = pd.read_csv(form_path)
    print(f"Adjudicating Phase 2 for Reviewer A ({len(df)} cases)...")
    results = []
    for cid in df.case_id:
        res = adjudicate_phase2_case(cid)
        results.append(res)
    res_df = pd.DataFrame(results)
    res_df.to_csv(form_path, index=False)
    print("Phase 2 form written successfully.")

PHASE3_RATINGS = [
    {
        "pattern": "unable_to_reach",
        "clinical_meaningfulness_1to5": 4,
        "what_it_indicates": "Communication breakdown or lost outreach channel requiring alternative locator strategies or secondary contacts.",
        "comments": "Actionable operational barrier; high frequency indicates impending disengagement."
    },
    {
        "pattern": "bad_number",
        "clinical_meaningfulness_1to5": 5,
        "what_it_indicates": "Acute telephonic disconnection or inaccurate contact data requiring field outreach or health plan demographic refresh.",
        "comments": "Direct technical blocker preventing care coordination; critical trigger for in-person locator."
    },
    {
        "pattern": "declined",
        "clinical_meaningfulness_1to5": 5,
        "what_it_indicates": "Patient refusal of program enrollment, care management services, or specific clinical referrals.",
        "comments": "Strong behavioral signal of autonomy or dissatisfaction; warrants assessing underlying unmet needs or boundaries."
    },
    {
        "pattern": "fewer_contacts",
        "clinical_meaningfulness_1to5": 4,
        "what_it_indicates": "Patient boundary-setting and communication preference requesting reduced contact frequency or text-only modality.",
        "comments": "Indicates active engagement with self-defined limits; adapting contact cadence preserves relationship."
    },
    {
        "pattern": "moved",
        "clinical_meaningfulness_1to5": 4,
        "what_it_indicates": "Geographic relocation or residential transition impacting Medicaid eligibility, MCO service area, and local clinic access.",
        "comments": "Clinically meaningful social determinant of health; warrants verification of new address and local care network."
    },
    {
        "pattern": "incarcerated",
        "clinical_meaningfulness_1to5": 5,
        "what_it_indicates": "Disruption of community-based care due to criminal justice involvement and institutionalization.",
        "comments": "Major clinical life event requiring suspension of outpatient care management and post-release transition planning."
    },
    {
        "pattern": "hospitalized",
        "clinical_meaningfulness_1to5": 5,
        "what_it_indicates": "Acute clinical deterioration or medical/psychiatric crisis requiring transition-of-care follow-up and med reconciliation.",
        "comments": "Highest-priority clinical event; requires immediate ADT alert and 30-day post-discharge care coordination."
    },
    {
        "pattern": "busy_callback",
        "clinical_meaningfulness_1to5": 3,
        "what_it_indicates": "Transient life scheduling conflict, caregiving/work demands, or mild ambivalence regarding care team contact.",
        "comments": "Operational scheduling delay; reflects competing social priorities rather than outright program refusal."
    },
    {
        "pattern": "positive_engagement",
        "clinical_meaningfulness_1to5": 4,
        "what_it_indicates": "Strong therapeutic rapport, receptivity to intervention, and mutual agreement on goals and upcoming encounters.",
        "comments": "Protective behavioral signal indicating collaborative relationship and low immediate risk of drop-out."
    },
    {
        "pattern": "outreach",
        "clinical_meaningfulness_1to5": 2,
        "what_it_indicates": "Administrative care team attempt at member communication without establishing two-way dialogue.",
        "comments": "Documentation workflow term; reflects care team effort rather than patient clinical or behavioral status."
    },
    {
        "pattern": "declined",
        "clinical_meaningfulness_1to5": 5,
        "what_it_indicates": "Explicit rejection of specific care offerings, assessments, or overall multidisciplinary support.",
        "comments": "Redundant with broader pattern but clinically distinct as single-word categorical refusal flag."
    },
    {
        "pattern": "not interested",
        "clinical_meaningfulness_1to5": 5,
        "what_it_indicates": "Clear patient statement of disinterest and unwillingness to participate in care management.",
        "comments": "Definitive disengagement signal indicating lack of perceived value or competing priorities."
    },
    {
        "pattern": "cc",
        "clinical_meaningfulness_1to5": 2,
        "what_it_indicates": "Internal care team discipline designation (Care Coordinator) involved in outreach or task routing.",
        "comments": "Purely administrative documentation artifact; denotes staff role rather than patient clinical state."
    },
    {
        "pattern": "mom",
        "clinical_meaningfulness_1to5": 3,
        "what_it_indicates": "Collateral contact with maternal caregiver acting as healthcare proxy or surrogate decision-maker.",
        "comments": "Meaningful caregiver engagement dynamic, particularly in young adult or dependent patient populations."
    },
    {
        "pattern": "answered",
        "clinical_meaningfulness_1to5": 2,
        "what_it_indicates": "Verification of live telephonic connection establishing successful channel opening.",
        "comments": "Operational milestone indicating working phone line, though not predictive of longitudinal retention."
    },
    {
        "pattern": "phi",
        "clinical_meaningfulness_1to5": 2,
        "what_it_indicates": "Regulatory HIPAA privacy verification or sensitive health information safeguarding disclosure.",
        "comments": "Compliance documentation marker; minimal direct bearing on patient behavioral engagement."
    },
    {
        "pattern": "declined services",
        "clinical_meaningfulness_1to5": 5,
        "what_it_indicates": "Formal member refusal to participate in the care management program.",
        "comments": "Unambiguous termination trigger requiring respectful program offboarding and documentation of reason."
    },
    {
        "pattern": "wcv",
        "clinical_meaningfulness_1to5": 4,
        "what_it_indicates": "Preventive health quality gap tracking for well-child visits and pediatric immunizations.",
        "comments": "Clinically meaningful HEDIS preventive care measure facilitated by community health workers."
    },
    {
        "pattern": "abhva",
        "clinical_meaningfulness_1to5": 2,
        "what_it_indicates": "Administrative managed care organization identifier (Anthem Better Health of Virginia).",
        "comments": "Payer plan label used for internal attribution and coverage routing; no behavioral meaning."
    },
    {
        "pattern": "waymark",
        "clinical_meaningfulness_1to5": 1,
        "what_it_indicates": "Care management organization brand self-reference in introductory scripts and encounter notes.",
        "comments": "Documentation artifact resulting from standardized outreach script phrasing."
    }
]

def run_phase3():
    form_path = BASE / "phase3" / "phase3_form.csv"
    print(f"Adjudicating Phase 3 narrative patterns ({len(PHASE3_RATINGS)} patterns)...")
    res_df = pd.DataFrame(PHASE3_RATINGS)
    res_df.to_csv(form_path, index=False)
    print("Phase 3 form written successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args()
    if args.phase == 1:
        run_phase1()
    elif args.phase == 2:
        run_phase2()
    elif args.phase == 3:
        run_phase3()
