"""
Vital Signs Clinical Analysis & Risk Stratification
Evaluates Blood Pressure, Heart Rate, Glucose, SpO2, and BMI.
"""

from typing import Dict, Any, Optional


class VitalsTracker:
    """
    Classifies vital signs according to standard medical guidelines (AHA, ADA, WHO).
    """

    @staticmethod
    def analyze_blood_pressure(systolic: int, diastolic: int) -> Dict[str, Any]:
        """AHA Guidelines for Blood Pressure Classification."""
        if systolic > 180 or diastolic > 120:
            category = "Hypertensive Crisis (EMERGENCY)"
            color = "#ef4444"
            status = "CRITICAL"
            advice = "Seek immediate emergency medical attention if accompanied by chest pain, shortness of breath, or numbness."
        elif systolic >= 140 or diastolic >= 90:
            category = "Stage 2 Hypertension"
            color = "#f97316"
            status = "HIGH"
            advice = "Consult your physician for prescription medication adjustment and lifestyle changes."
        elif (130 <= systolic <= 139) or (80 <= diastolic <= 89):
            category = "Stage 1 Hypertension"
            color = "#eab308"
            status = "ELEVATED"
            advice = "Adopt dietary changes (low sodium DASH diet), regular exercise, and stress reduction."
        elif 120 <= systolic <= 129 and diastolic < 80:
            category = "Elevated Blood Pressure"
            color = "#38bdf8"
            status = "BORDERLINE"
            advice = "Monitor regularly and maintain active physical lifestyle."
        elif systolic < 90 or diastolic < 60:
            category = "Hypotension (Low BP)"
            color = "#a855f7"
            status = "LOW"
            advice = "Ensure adequate hydration and salt intake; see a doctor if experiencing dizziness or fainting."
        else:
            category = "Normal Blood Pressure"
            color = "#10b981"
            status = "NORMAL"
            advice = "Optimal cardiovascular health. Keep up your healthy lifestyle habits."

        return {
            "reading": f"{systolic}/{diastolic} mmHg",
            "category": category,
            "status": status,
            "color": color,
            "clinical_advice": advice,
        }

    @staticmethod
    def analyze_heart_rate(bpm: int) -> Dict[str, Any]:
        """Resting Heart Rate Classification."""
        if bpm > 100:
            category = "Tachycardia (High Heart Rate)"
            status = "ELEVATED"
            color = "#f97316"
            advice = "May be caused by stress, caffeine, fever, dehydration, or cardiac arrhythmia. Rest and recheck."
        elif bpm < 60:
            category = "Bradycardia (Low Heart Rate)"
            status = "LOW"
            color = "#38bdf8"
            advice = "Normal in well-trained athletes, but if accompanied by dizziness or fatigue, medical review is advised."
        else:
            category = "Normal Resting Heart Rate"
            status = "NORMAL"
            color = "#10b981"
            advice = "Heart rate is within optimal resting range (60-100 BPM)."

        return {
            "reading": f"{bpm} BPM",
            "category": category,
            "status": status,
            "color": color,
            "clinical_advice": advice,
        }

    @staticmethod
    def analyze_glucose(mg_dl: float, test_type: str = "fasting") -> Dict[str, Any]:
        """ADA Blood Glucose Classification."""
        if mg_dl < 70:
            category = "Hypoglycemia (Low Blood Sugar)"
            status = "CRITICAL"
            color = "#ef4444"
            advice = "Consume 15-20g of fast-acting carbohydrates (fruit juice, candy) and retest in 15 minutes."
        elif test_type.lower() == "fasting":
            if mg_dl >= 126:
                category = "Diabetes Range (Fasting)"
                status = "HIGH"
                color = "#f97316"
                advice = "Consult your physician for confirmation (HbA1c test) and personalized treatment."
            elif 100 <= mg_dl <= 125:
                category = "Pre-diabetes (Impaired Fasting Glucose)"
                status = "ELEVATED"
                color = "#eab308"
                advice = "Lifestyle interventions (diet and physical activity) can effectively prevent progression to Type 2 diabetes."
            else:
                category = "Normal Fasting Glucose"
                status = "NORMAL"
                color = "#10b981"
                advice = "Healthy glucose metabolism."
        else:
            # Random / Post-prandial
            if mg_dl >= 200:
                category = "Elevated Blood Glucose"
                status = "HIGH"
                color = "#f97316"
                advice = "High reading; schedule a medical consultation."
            else:
                category = "Normal Post-Meal Glucose"
                status = "NORMAL"
                color = "#10b981"
                advice = "Within normal post-meal limits."

        return {
            "reading": f"{mg_dl} mg/dL ({test_type})",
            "category": category,
            "status": status,
            "color": color,
            "clinical_advice": advice,
        }

    @staticmethod
    def analyze_spo2(percent: int) -> Dict[str, Any]:
        """Oxygen Saturation Classification."""
        if percent < 90:
            category = "Critical Hypoxemia"
            status = "CRITICAL"
            color = "#ef4444"
            advice = "Severely low blood oxygen. Seek emergency medical care immediately."
        elif percent <= 94:
            category = "Mild to Moderate Hypoxia"
            status = "WARNING"
            color = "#f97316"
            advice = "Lower than normal oxygen level. Consult a doctor or visit urgent care."
        else:
            category = "Normal Oxygen Saturation"
            status = "NORMAL"
            color = "#10b981"
            advice = "Healthy oxygenation level (95-100%)."

        return {
            "reading": f"{percent}%",
            "category": category,
            "status": status,
            "color": color,
            "clinical_advice": advice,
        }
