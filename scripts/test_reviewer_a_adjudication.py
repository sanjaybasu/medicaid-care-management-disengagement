import pathlib
import pandas as pd
import pytest

BASE = pathlib.Path(__file__).resolve().parent.parent / "data_cache" / "physician_review"

def test_phase1_reviewer_a_schema_and_values():
    p1 = BASE / "phase1" / "reviewer_A_form.csv"
    assert p1.exists(), "Phase 1 form missing"
    df = pd.read_csv(p1)
    assert len(df) == 143, f"Expected 143 rows, found {len(df)}"
    assert (df.reviewer == "A").all()

    valid_q1 = {"engaged", "ambivalent", "declining", "unreachable", "administrative"}
    valid_q4 = {"low", "moderate", "high"}
    valid_q5 = {"continue", "fewer_contacts", "stop", "none"}

    for idx, r in df.iterrows():
        assert r.q1_engagement_status in valid_q1, f"Row {idx} ({r.case_id}) invalid q1: {r.q1_engagement_status}"
        assert int(r.q2_prognosis_1to5) in {1, 2, 3, 4, 5}, f"Row {idx} ({r.case_id}) invalid q2: {r.q2_prognosis_1to5}"
        for col in [
            "q3_open_need_medication",
            "q3_open_need_referral_or_appointment",
            "q3_open_need_uncontrolled_condition",
            "q3_open_need_social_crisis",
            "q3_open_need_behavioral_health",
            "q3_no_open_need",
        ]:
            assert int(r[col]) in {0, 1}, f"Row {idx} ({r.case_id}) invalid {col}: {r[col]}"

        needs_sum = (
            int(r.q3_open_need_medication)
            + int(r.q3_open_need_referral_or_appointment)
            + int(r.q3_open_need_uncontrolled_condition)
            + int(r.q3_open_need_social_crisis)
            + int(r.q3_open_need_behavioral_health)
        )
        if needs_sum == 0:
            assert int(r.q3_no_open_need) == 1, f"Row {idx} ({r.case_id}): no needs indicated but q3_no_open_need is not 1"
        else:
            assert int(r.q3_no_open_need) == 0, f"Row {idx} ({r.case_id}): open needs present but q3_no_open_need is 1"

        assert r.q4_harm_if_lost in valid_q4, f"Row {idx} ({r.case_id}) invalid q4: {r.q4_harm_if_lost}"
        assert r.q5_stated_intent in valid_q5, f"Row {idx} ({r.case_id}) invalid q5: {r.q5_stated_intent}"
        assert isinstance(r.comments, str) and len(r.comments.strip()) > 5, f"Row {idx} ({r.case_id}) comments too short"
        assert not any(ph in str(r.comments).lower() for ph in ["todo", "tbd", "placeholder", "xxx"]), f"Row {idx} contains placeholder"

def test_phase2_reviewer_a_schema_and_values():
    p2 = BASE / "phase2" / "reviewer_A_phase2_form.csv"
    assert p2.exists(), "Phase 2 form missing"
    df = pd.read_csv(p2)
    assert len(df) == 41, f"Expected 41 rows, found {len(df)}"
    assert (df.reviewer == "A").all()

    valid_reasons = {
        "needs_met",
        "declined_or_dissatisfied",
        "lost_contact",
        "life_event",
        "administrative_or_coverage",
        "cannot_determine",
    }
    for idx, r in df.iterrows():
        assert r.reason in valid_reasons, f"Row {idx} ({r.case_id}) invalid reason: {r.reason}"
        assert int(r.confidence_1to5) in {1, 2, 3, 4, 5}, f"Row {idx} ({r.case_id}) invalid confidence: {r.confidence_1to5}"
        assert isinstance(r.comments, str) and len(r.comments.strip()) > 5, f"Row {idx} ({r.case_id}) comments too short"
        assert not any(ph in str(r.comments).lower() for ph in ["todo", "tbd", "placeholder", "xxx"]), f"Row {idx} contains placeholder"

def test_phase3_schema_and_values():
    p3 = BASE / "phase3" / "phase3_form.csv"
    assert p3.exists(), "Phase 3 form missing"
    df = pd.read_csv(p3)
    assert len(df) == 20, f"Expected 20 rows, found {len(df)}"
    for idx, r in df.iterrows():
        assert int(r.clinical_meaningfulness_1to5) in {1, 2, 3, 4, 5}
        assert isinstance(r.what_it_indicates, str) and len(r.what_it_indicates.strip()) > 5
        assert not any(ph in str(r.what_it_indicates).lower() for ph in ["todo", "tbd", "placeholder", "xxx"])

def test_other_reviewers_unaltered():
    for f in [
        BASE / "phase1" / "reviewer_B_form.csv",
        BASE / "phase1" / "reviewer_C_form.csv",
        BASE / "phase2" / "reviewer_B_phase2_form.csv",
        BASE / "phase2" / "reviewer_C_phase2_form.csv",
    ]:
        df = pd.read_csv(f)
        if "q1_engagement_status" in df.columns:
            assert df["q1_engagement_status"].isna().all(), f"{f.name} was modified!"
        elif "reason" in df.columns:
            assert df["reason"].isna().all(), f"{f.name} was modified!"
