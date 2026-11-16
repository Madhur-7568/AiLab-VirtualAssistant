"""
Built-in Clinical Medical Knowledge Base
Provides evidence-based healthcare facts, symptom etiologies, drug interactions,
first-aid steps, and emergency triage definitions.
"""

from typing import Dict, List, Any, Optional
import re


class MedicalKnowledgeBase:
    """
    Curated clinical repository of symptoms, common medicines, first aid, and red-flag conditions.
    """

    EMERGENCY_RED_FLAGS = [
        "chest pain", "pressure in chest", "left arm pain", "heart attack",
        "difficulty breathing", "cannot breathe", "shortness of breath", "severe asthma",
        "stroke", "facial drooping", "slurred speech", "sudden numbness", "loss of vision",
        "heavy bleeding", "unconscious", "unresponsive", "severe allergic reaction",
        "anaphylaxis", "throat swelling", "coughing blood", "seizure", "overdose"
    ]

    SYMPTOM_DATABASE: Dict[str, Dict[str, Any]] = {
        "fever": {
            "name": "Fever (Pyrexia)",
            "causes": ["Viral infection (common cold, flu)", "Bacterial infection (strep, UTI)", "Inflammatory condition", "Heat exhaustion"],
            "self_care": ["Stay hydrated with water, oral rehydration salts, or clear broths.", "Get adequate rest.", "Use a cool damp washcloth on the forehead.", "Dress in lightweight, breathable clothing."],
            "otc_options": ["Paracetamol (Acetaminophen) 500mg-1000mg every 4-6 hours (max 4000mg/day)", "Ibuprofen 200mg-400mg with food every 6-8 hours (if no kidney/stomach issues)"],
            "red_flags": "Fever above 103°F (39.4°C), stiff neck, confusion, rash, or lasting more than 3 consecutive days.",
        },
        "headache": {
            "name": "Headache (Cephalea / Migraine / Tension)",
            "causes": ["Tension / stress", "Dehydration", "Lack of sleep / eye strain", "Migraine", "Sinusitis"],
            "self_care": ["Rest in a quiet, dark room.", "Drink 500ml-1L of water.", "Apply a cold or warm compress to forehead or neck.", "Limit screen time."],
            "otc_options": ["Paracetamol (Acetaminophen)", "Ibuprofen", "Naproxen"],
            "red_flags": "Sudden 'thunderclap' headache (worst headache of your life), accompanied by fever, neck stiffness, confusion, or weakness on one side.",
        },
        "cough": {
            "name": "Cough (Acute / Chronic)",
            "causes": ["Upper respiratory viral infection", "Allergies / post-nasal drip", "Acid reflux (GERD)", "Bronchitis", "Asthma"],
            "self_care": ["Drink warm liquids (honey and lemon water; do not give honey to infants under 1 year).", "Use a steam inhaler or humidifier.", "Gargle with warm salt water."],
            "otc_options": ["Dextromethorphan for dry tickly cough", "Guaifenesin for productive/phlegm cough", "Antihistamines for allergic cough"],
            "red_flags": "Coughing up blood or rust-colored sputum, high fever, severe shortness of breath, or cough lasting longer than 3-4 weeks.",
        },
        "sore throat": {
            "name": "Sore Throat (Pharyngitis)",
            "causes": ["Viral pharyngitis (most common)", "Streptococcal bacterial pharyngitis (Strep throat)", "Dry air", "Allergies"],
            "self_care": ["Warm salt water gargles (1/2 tsp salt in 1 cup warm water) 3-4 times daily.", "Warm herbal teas with honey.", "Throat lozenges or hard candies to stimulate saliva."],
            "otc_options": ["Paracetamol or Ibuprofen for pain and inflammation", "Antiseptic/anesthetic throat lozenges or sprays"],
            "red_flags": "Difficulty swallowing saliva, inability to open mouth fully (trismus), breathing difficulty, or rash.",
        },
        "stomach pain": {
            "name": "Abdominal Discomfort / Stomach Ache",
            "causes": ["Indigestion / Gastritis", "Gas and bloating", "Gastroenteritis (stomach flu)", "Food intolerance", "Constipation"],
            "self_care": ["Eat bland foods (BRAT diet: Bananas, Rice, Applesauce, Toast).", "Avoid spicy, greasy, acidic, or caffeinated food/drinks.", "Sip peppermint or chamomile tea.", "Use a warm heating pad on the abdomen."],
            "otc_options": ["Antacids (Calcium carbonate, Magnesium hydroxide) for heartburn", "Simethicone for bloating/gas", "Oral rehydration solution if vomiting/diarrhea occurs"],
            "red_flags": "Severe sharp pain in the lower right abdomen (possible appendicitis), vomiting blood, black tarry stools, high fever, or rigid abdomen.",
        },
        "migraine": {
            "name": "Migraine",
            "causes": ["Neurovascular headache triggered by stress, hormonal changes, bright lights, dehydration, or dietary triggers"],
            "self_care": ["Rest in a pitch-black, silent room.", "Apply an ice pack to temples.", "Drink fluids with electrolytes.", "Practice controlled slow breathing."],
            "otc_options": ["NSAIDs (Ibuprofen, Naproxen) taken at early onset", "Paracetamol + Caffeine combination"],
            "red_flags": "Visual loss, speech difficulty, confusion, or headache accompanied by fever and neck stiffness.",
        },
        "blood pressure": {
            "name": "Hypertension / High Blood Pressure",
            "causes": ["High sodium diet, stress, genetics, lack of exercise, obesity, kidney issues"],
            "self_care": ["Adopt the DASH diet (rich in fruits, vegetables, low-fat dairy).", "Reduce daily sodium intake to < 2,300mg (ideally 1,500mg).", "Engage in 150 minutes of moderate aerobic exercise per week.", "Limit alcohol and quit smoking.", "Practice stress-reduction meditation."],
            "otc_options": ["Blood pressure medications MUST be prescribed by a physician (e.g., ACE inhibitors, ARBs, Calcium Channel Blockers). Never self-medicate."],
            "red_flags": "Systolic > 180 mmHg or Diastolic > 120 mmHg accompanied by chest pain, shortness of breath, back pain, numbness/weakness, or change in vision (Hypertensive Emergency).",
        },
        "diabetes": {
            "name": "Diabetes / Blood Sugar Management",
            "causes": ["Insulin resistance (Type 2) or autoimmune destruction of pancreatic beta cells (Type 1)"],
            "self_care": ["Monitor blood glucose regularly.", "Eat low-glycemic index carbohydrates with high dietary fiber.", "Maintain consistent meal timings.", "Engage in post-meal light walking (15-20 mins)."],
            "otc_options": ["Prescribed insulin or oral hypoglycemics (e.g., Metformin). Keep glucose tablets or juice handy for hypoglycemia (< 70 mg/dL)."],
            "red_flags": "Extreme confusion, fruity breath odor, rapid deep breathing, vomiting, or blood sugar > 300 mg/dL with ketones (DKA risk).",
        },
    }

    MEDICATIONS_DATABASE: Dict[str, Dict[str, Any]] = {
        "paracetamol": {
            "generic": "Acetaminophen / Paracetamol",
            "class": "Analgesic & Antipyretic",
            "indication": "Mild to moderate pain and fever reduction.",
            "usual_dose": "Adults: 500mg to 1000mg every 4 to 6 hours as needed. Maximum daily dose is 4000mg (3000mg for prolonged use or elderly).",
            "precautions": "Avoid taking multiple cold/flu medications containing paracetamol simultaneously. Avoid chronic heavy alcohol use due to liver hepatotoxicity risk.",
            "interactions": "Safe with most drugs; caution with Warfarin in high continuous doses.",
        },
        "ibuprofen": {
            "generic": "Ibuprofen",
            "class": "Non-Steroidal Anti-Inflammatory Drug (NSAID)",
            "indication": "Inflammatory pain, swelling, menstrual cramps, toothache, fever.",
            "usual_dose": "Adults: 200mg to 400mg every 6 to 8 hours with food or milk. Maximum OTC daily dose is 1200mg (up to 2400mg under prescription).",
            "precautions": "Take with food to protect stomach lining. Contraindicated in active peptic ulcer disease, severe heart failure, severe renal impairment, and late pregnancy.",
            "interactions": "Do not combine with other NSAIDs (Aspirin, Naproxen) or blood thinners without doctor approval.",
        },
        "cetirizine": {
            "generic": "Cetirizine Hydrochloride",
            "class": "Second-generation Antihistamine",
            "indication": "Allergic rhinitis, hay fever, hives, itchy eyes, sneezing.",
            "usual_dose": "Adults: 10mg once daily (preferably in the evening).",
            "precautions": "May cause mild drowsiness in some individuals. Avoid alcohol while taking.",
            "interactions": "CNS depressants and sedatives may increase drowsiness.",
        },
        "omeprazole": {
            "generic": "Omeprazole",
            "class": "Proton Pump Inhibitor (PPI)",
            "indication": "Acid reflux, GERD, heartburn, stomach ulcers.",
            "usual_dose": "20mg once daily, 30-60 minutes before breakfast on an empty stomach.",
            "precautions": "Intended for short-term courses (14 days OTC). Consult physician if symptoms persist.",
            "interactions": "May interact with Clopidogrel, Diazepam, and iron supplements absorption.",
        },
    }

    @classmethod
    def check_emergency(cls, text: str) -> Optional[str]:
        """Detect critical emergency keywords in user input."""
        lower = text.lower()
        for red_flag in cls.EMERGENCY_RED_FLAGS:
            if re.search(r"\b" + re.escape(red_flag) + r"\b", lower):
                return red_flag
        return None

    @classmethod
    def match_symptom(cls, text: str) -> Optional[Dict[str, Any]]:
        """Match common symptom from knowledge base."""
        lower = text.lower()
        for key, data in cls.SYMPTOM_DATABASE.items():
            if key in lower or data["name"].lower() in lower:
                return data
        return None

    @classmethod
    def match_medication(cls, text: str) -> Optional[Dict[str, Any]]:
        """Match common medication query."""
        lower = text.lower()
        for key, data in cls.MEDICATIONS_DATABASE.items():
            if key in lower or data["generic"].lower() in lower:
                return data
        return None

    @classmethod
    def generate_expert_response(cls, query: str) -> str:
        """
        Generate structured, clinical-grade medical response using embedded knowledge base.
        """
        # 1. Check for critical medical emergencies
        flag = cls.check_emergency(query)
        if flag:
            return (
                f"🚨 **CRITICAL MEDICAL ALERT: IMMEDIATE ATTENTION REQUIRED**\n\n"
                f"You mentioned symptoms related to **'{flag}'**, which can be a sign of a life-threatening medical emergency.\n\n"
                f"### Immediate Actions:\n"
                f"1. **Call Emergency Services Immediately:** Dial **911** (US), **112** (Europe/India), or your local emergency medical number.\n"
                f"2. **Do not drive yourself:** Request an ambulance so paramedics can initiate emergency care on the way.\n"
                f"3. **If chest pain / heart attack is suspected:** Sit down, stay calm, loosen tight clothing, and if you are not allergic and have medical approval, chew one standard adult aspirin (325mg).\n"
                f"4. **Notify someone nearby:** Let a family member, colleague, or neighbor know your condition immediately.\n\n"
                f"*Disclaimer: This AI cannot diagnose or treat acute emergencies. Seek emergency medical care immediately.*"
            )

        # 2. Check for medication inquiry
        med = cls.match_medication(query)
        if med:
            return (
                f"💊 **Medication Information: {med['generic']}**\n\n"
                f"- **Drug Class:** {med['class']}\n"
                f"- **Common Indications:** {med['indication']}\n"
                f"- **Standard Dosage Guidance:** {med['usual_dose']}\n"
                f"- **Important Precautions:** {med['precautions']}\n"
                f"- **Key Interactions:** {med['interactions']}\n\n"
                f"💡 *Clinical Tip:* Always verify with your pharmacist or prescribing doctor, especially if you have liver, kidney, or cardiovascular conditions.\n\n"
                f"*Disclaimer: Informational only. Do not alter prescribed medications without consulting your healthcare provider.*"
            )

        # 3. Check for specific symptom
        symptom = cls.match_symptom(query)
        if symptom:
            causes_list = "\n".join([f"  • {c}" for c in symptom["causes"]])
            self_care_list = "\n".join([f"  • {sc}" for sc in symptom["self_care"]])
            otc_list = "\n".join([f"  • {otc}" for otc in symptom["otc_options"]])
            return (
                f"🩺 **Clinical Overview: {symptom['name']}**\n\n"
                f"### 🔍 Common Underlying Causes:\n{causes_list}\n\n"
                f"### 🌿 Recommended Self-Care & Recovery Steps:\n{self_care_list}\n\n"
                f"### 💊 Common Over-The-Counter Options:\n{otc_list}\n\n"
                f"### ⚠️ Red Flags (When to See a Doctor Promptly):\n{symptom['red_flags']}\n\n"
                f"*Clinical Note: If symptoms persist beyond 48-72 hours, worsen significantly, or do not respond to self-care, schedule a clinical consultation.*"
            )

        # 4. General healthcare / lifestyle / doubt fallback
        return (
            f"🩺 **Healthcare Virtual Assistant Guidance**\n\n"
            f"Thank you for sharing your health query: *\"{query}\"*\n\n"
            f"### Key Medical Considerations:\n"
            f"1. **Symptom Duration & Severity:** Note when your symptoms started, what triggers them, and whether they are mild, moderate, or acute.\n"
            f"2. **Hydration & Rest:** Ensure adequate fluid intake (water, clear broths) and prioritize 7-9 hours of restorative sleep.\n"
            f"3. **Vital Monitoring:** If applicable, monitor your temperature, blood pressure, and pulse rate.\n"
            f"4. **Medication Safety:** Do not mix pain relievers without checking their active ingredients to prevent accidental overdose.\n\n"
            f"### When to Contact a Healthcare Professional:\n"
            f"- If your symptoms are interfering with daily activities or work.\n"
            f"- If you experience severe pain, high persistent fever, or unmanageable fatigue.\n"
            f"- If you are pregnant, elderly, or have chronic conditions (diabetes, hypertension, asthma).\n\n"
            f"💬 *Feel free to describe specific symptoms (e.g. fever, headache, blood pressure reading) or use the Symptom Triage tool in the top tab!*"
        )
