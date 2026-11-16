"""
Unit tests for Vital Signs Classification
"""

from src.healthcare.vitals_tracker import VitalsTracker


def test_blood_pressure_normal():
    res = VitalsTracker.analyze_blood_pressure(118, 76)
    assert res["status"] == "NORMAL"
    assert "Normal" in res["category"]


def test_blood_pressure_hypertensive_crisis():
    res = VitalsTracker.analyze_blood_pressure(190, 125)
    assert res["status"] == "CRITICAL"
    assert "Crisis" in res["category"]


def test_heart_rate_tachycardia():
    res = VitalsTracker.analyze_heart_rate(115)
    assert res["status"] == "ELEVATED"
    assert "Tachycardia" in res["category"]


def test_glucose_fasting_diabetes():
    res = VitalsTracker.analyze_glucose(135, test_type="fasting")
    assert res["status"] == "HIGH"
    assert "Diabetes" in res["category"]


def test_spo2_critical():
    res = VitalsTracker.analyze_spo2(88)
    assert res["status"] == "CRITICAL"
    assert "Hypoxemia" in res["category"]
