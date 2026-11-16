"""
Unit tests for Healthcare LLM Engine and Clinical Knowledge Base
"""

from src.healthcare.llm_engine import HealthcareLLMEngine
from src.healthcare.medical_knowledge import MedicalKnowledgeBase


def test_emergency_red_flag_detection():
    # Should detect emergency
    assert MedicalKnowledgeBase.check_emergency("I have severe chest pain and left arm numbness") is not None
    assert MedicalKnowledgeBase.check_emergency("Patient is unconscious and having difficulty breathing") is not None
    # Normal query should not trigger emergency
    assert MedicalKnowledgeBase.check_emergency("What are good home remedies for a mild cold?") is None


def test_knowledge_base_symptom_match():
    res = MedicalKnowledgeBase.match_symptom("I have had a high fever for two days")
    assert res is not None
    assert "Fever" in res["name"]


def test_knowledge_base_medication_match():
    res = MedicalKnowledgeBase.match_medication("Can I take paracetamol for pain?")
    assert res is not None
    assert "Acetaminophen" in res["generic"] or "Paracetamol" in res["generic"]


def test_llm_engine_fallback_response():
    engine = HealthcareLLMEngine(api_key=None)
    result = engine.generate_response("What should I do for a mild headache?")

    assert result["is_emergency"] is False
    assert "Headache" in result["response"]
    assert "Clinical Expert Knowledge Engine" in result["model_used"]


def test_llm_engine_emergency_override():
    engine = HealthcareLLMEngine(api_key=None)
    result = engine.generate_response("I have unbearable crushing chest pain and shortness of breath")

    assert result["is_emergency"] is True
    assert "CRITICAL MEDICAL ALERT" in result["response"]
    assert "911" in result["response"] or "112" in result["response"]
