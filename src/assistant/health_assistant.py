"""
Healthcare Virtual Assistant Orchestrator
Integrates LLM responses, symptom triage, vitals tracking, and conversational memory.
"""

from typing import Dict, Any, List, Optional
import datetime
from src.healthcare.llm_engine import HealthcareLLMEngine
from src.healthcare.symptom_triage import SymptomTriage
from src.healthcare.vitals_tracker import VitalsTracker


class HealthAssistant:
    """
    Main virtual assistant pipeline for healthcare doubts, symptom triage, and vitals assessment.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.llm_engine = HealthcareLLMEngine(api_key=api_key)
        self.triage_engine = SymptomTriage()
        self.vitals_engine = VitalsTracker()
        self.conversation_history: List[Dict[str, str]] = []

    def chat(self, user_message: str, api_key_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Process user health doubt or conversation through the LLM pipeline.
        """
        llm_result = self.llm_engine.generate_response(
            query=user_message,
            conversation_history=self.conversation_history,
            api_key_override=api_key_override,
        )

        # Store in conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": llm_result["response"]})

        return {
            "query": user_message,
            "response": llm_result["response"],
            "model_used": llm_result["model_used"],
            "is_emergency": llm_result["is_emergency"],
            "disclaimer": llm_result["disclaimer"],
            "timestamp": datetime.datetime.now().isoformat(),
        }

    def check_symptoms(
        self,
        symptom: str,
        severity: int = 4,
        duration: int = 1,
        additional: Optional[List[str]] = None,
        chronic: bool = False,
        age: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run clinical triage assessment.
        """
        return self.triage_engine.evaluate(
            primary_symptom=symptom,
            severity_1_to_10=severity,
            duration_days=duration,
            additional_symptoms=additional,
            has_chronic_illness=chronic,
            age=age,
        )

    def analyze_vitals(
        self,
        systolic: Optional[int] = None,
        diastolic: Optional[int] = None,
        heart_rate: Optional[int] = None,
        blood_glucose: Optional[float] = None,
        glucose_type: str = "fasting",
        spo2: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate full panel of vital signs.
        """
        results: Dict[str, Any] = {}
        if systolic is not None and diastolic is not None:
            results["blood_pressure"] = self.vitals_engine.analyze_blood_pressure(systolic, diastolic)
        if heart_rate is not None:
            results["heart_rate"] = self.vitals_engine.analyze_heart_rate(heart_rate)
        if blood_glucose is not None:
            results["blood_glucose"] = self.vitals_engine.analyze_glucose(blood_glucose, glucose_type)
        if spo2 is not None:
            results["spo2"] = self.vitals_engine.analyze_spo2(spo2)

        return results
