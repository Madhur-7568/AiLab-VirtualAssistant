"""
Educational Demo Script: HealthPulse AI Healthcare Virtual Assistant
--------------------------------------------------------------------
Demonstrates:
1. Conversational Healthcare LLM / Clinical Expert Query Answering
2. Clinical Symptom Triage (Severity, Duration, Red-Flag Protocol)
3. Vital Signs Assessment & Cardiovascular Risk Stratification
"""

from src.assistant.health_assistant import HealthAssistant


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def demo_healthcare_assistant():
    assistant = HealthAssistant()

    # 1. Healthcare Queries & Medical Doubts
    print_section("1. Healthcare Chatbot & Medical Doubt Resolution")
    queries = [
        "What are the best home care steps for a high fever?",
        "Can I take Paracetamol with Ibuprofen?",
        "I have severe chest pain and left arm numbness",
    ]

    for q in queries:
        print(f"\n>>> USER QUERY: \"{q}\"")
        res = assistant.chat(q)
        print(f"    [Model Used]: {res['model_used']}")
        print(f"    [Emergency Alert]: {res['is_emergency']}")
        print(f"\n{res['response']}\n")
        print("-" * 50)

    # 2. Clinical Symptom Triage
    print_section("2. Clinical Symptom Triage Assessment")
    triage_cases = [
        {"symptom": "Mild sore throat and runny nose", "severity": 2, "duration": 1},
        {"symptom": "Persistent cough with high fever", "severity": 7, "duration": 8},
        {"symptom": "Difficulty breathing and chest tightness", "severity": 9, "duration": 1},
    ]

    for case in triage_cases:
        triage = assistant.check_symptoms(
            symptom=case["symptom"],
            severity=case["severity"],
            duration=case["duration"],
        )
        print(f"\nSymptom: '{case['symptom']}' (Severity: {case['severity']}/10, Duration: {case['duration']} days)")
        print(f"  • Triage Level:      {triage['badge']}")
        print(f"  • Recommended Action:{triage['recommended_action']}")
        print(f"  • Timeframe:         {triage['timeframe']}")
        print(f"  • Clinical Rationale:{triage['clinical_rationale']}")

    # 3. Vital Signs Analysis
    print_section("3. Vital Signs & Biomarker Classification")
    vitals = assistant.analyze_vitals(
        systolic=148,
        diastolic=94,
        heart_rate=105,
        blood_glucose=140,
        glucose_type="fasting",
        spo2=93,
    )

    print("Patient Vitals Panel:")
    for metric, data in vitals.items():
        print(f"  • {metric.replace('_', ' ').title():<18}: {data['reading']} [{data['category']}] -> {data['clinical_advice']}")


if __name__ == "__main__":
    demo_healthcare_assistant()
    print("\n[SUCCESS] HealthPulse AI Healthcare Virtual Assistant demo finished successfully!\n")
