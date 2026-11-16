"""
Unit tests for Clinical Symptom Triage
"""

from src.healthcare.symptom_triage import SymptomTriage


def test_triage_emergency_symptom():
    triage = SymptomTriage.evaluate(
        primary_symptom="chest pain",
        severity_1_to_10=8,
        duration_days=1,
    )
    assert triage["triage_level"] == "EMERGENCY"
    assert "Immediate" in triage["timeframe"]


def test_triage_severe_pain_level():
    triage = SymptomTriage.evaluate(
        primary_symptom="abdominal cramp",
        severity_1_to_10=10,
        duration_days=1,
    )
    assert triage["triage_level"] == "EMERGENCY"


def test_triage_urgent_care():
    triage = SymptomTriage.evaluate(
        primary_symptom="migraine with vomiting",
        severity_1_to_10=7,
        duration_days=2,
    )
    assert triage["triage_level"] == "URGENT_CARE"


def test_triage_self_care():
    triage = SymptomTriage.evaluate(
        primary_symptom="mild runny nose",
        severity_1_to_10=2,
        duration_days=1,
    )
    assert triage["triage_level"] == "SELF_CARE"
