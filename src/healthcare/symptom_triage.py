"""
Clinical Symptom Triage System
Evaluates symptoms, severity scales, duration, and patient risk factors
to determine safe triage recommendations and next steps.
"""

from typing import Dict, List, Any, Optional
from src.healthcare.medical_knowledge import MedicalKnowledgeBase


class SymptomTriage:
    """
    Automated Clinical Triage Engine based on emergency medicine decision algorithms.
    """

    TRIAGE_LEVELS = {
        "EMERGENCY": {
            "title": "Level 1: Emergency Medical Attention Required",
            "badge": "EMERGENCY (Immediate 911 / 112)",
            "color": "#ef4444",
            "timeframe": "Immediate (within minutes)",
            "action": "Proceed to the nearest Emergency Department or call emergency medical services immediately.",
        },
        "URGENT_CARE": {
            "title": "Level 2: Urgent Care Evaluation Needed",
            "badge": "URGENT CARE",
            "color": "#f97316",
            "timeframe": "Within 12 to 24 hours",
            "action": "Visit an Urgent Care clinic or book a same-day appointment with your physician.",
        },
        "ROUTINE_CLINIC": {
            "title": "Level 3: Routine Clinical Consultation",
            "badge": "ROUTINE CLINIC",
            "color": "#eab308",
            "timeframe": "Within 2 to 3 days",
            "action": "Schedule an in-person or telehealth appointment with your primary care doctor.",
        },
        "SELF_CARE": {
            "title": "Level 4: Home Monitoring & Self-Care",
            "badge": "SELF-CARE",
            "color": "#10b981",
            "timeframe": "Monitor over next 48-72 hours",
            "action": "Follow rest, hydration, and over-the-counter self-care measures. Re-evaluate if symptoms escalate.",
        },
    }

    @classmethod
    def evaluate(
        cls,
        primary_symptom: str,
        severity_1_to_10: int = 3,
        duration_days: int = 1,
        additional_symptoms: Optional[List[str]] = None,
        has_chronic_illness: bool = False,
        age: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute triage level based on symptom profile.
        """
        additional_symptoms = additional_symptoms or []
        combined_text = f"{primary_symptom} " + " ".join(additional_symptoms)

        # 1. Check for emergency red flags
        emergency_trigger = MedicalKnowledgeBase.check_emergency(combined_text)
        if emergency_trigger or severity_1_to_10 >= 9:
            level_key = "EMERGENCY"
            reason = f"Detected acute red-flag symptom: '{emergency_trigger or 'Extreme Pain Level (9-10/10)'}'"

        # 2. Urgent care check
        elif severity_1_to_10 >= 7 or (duration_days > 7 and severity_1_to_10 >= 5) or (has_chronic_illness and severity_1_to_10 >= 6):
            level_key = "URGENT_CARE"
            reason = f"High severity score ({severity_1_to_10}/10) or prolonged symptom duration ({duration_days} days)."

        # 3. Routine clinic check
        elif duration_days >= 3 or severity_1_to_10 >= 4 or has_chronic_illness:
            level_key = "ROUTINE_CLINIC"
            reason = f"Persistent moderate symptoms ({duration_days} days) warranting doctor evaluation."

        # 4. Self-care check
        else:
            level_key = "SELF_CARE"
            reason = "Mild symptom presentation suitable for home management and close monitoring."

        details = cls.TRIAGE_LEVELS[level_key]

        return {
            "triage_level": level_key,
            "badge": details["badge"],
            "title": details["title"],
            "color": details["color"],
            "timeframe": details["timeframe"],
            "recommended_action": details["action"],
            "clinical_rationale": reason,
            "input_summary": {
                "primary_symptom": primary_symptom,
                "severity": f"{severity_1_to_10}/10",
                "duration_days": duration_days,
                "has_chronic_conditions": has_chronic_illness,
                "age": age,
            },
        }
