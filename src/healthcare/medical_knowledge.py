"""
Built-in Clinical Medical Knowledge Base
Provides evidence-based healthcare facts, symptom etiologies, drug interactions,
first-aid steps, and emergency triage definitions with bilingual support (English & Hindi/Hinglish).
"""

from typing import Dict, List, Any, Optional
import re


class MedicalKnowledgeBase:
    """
    Curated clinical repository of symptoms, common medicines, first aid, and red-flag conditions.
    Works completely offline with zero API keys required.
    """

    EMERGENCY_RED_FLAGS = [
        "chest pain", "pressure in chest", "left arm pain", "heart attack", "heart stroke",
        "difficulty breathing", "cannot breathe", "shortness of breath", "severe asthma", "sans lene me takleef",
        "stroke", "facial drooping", "slurred speech", "sudden numbness", "loss of vision",
        "heavy bleeding", "unconscious", "behosh", "unresponsive", "severe allergic reaction",
        "anaphylaxis", "throat swelling", "coughing blood", "seizure", "overdose"
    ]

    SYMPTOM_DATABASE: Dict[str, Dict[str, Any]] = {
        "fever": {
            "name": "Fever (Pyrexia / Bukhar)",
            "aliases": ["fever", "bukhar", "high temperature", "chills", "garam"],
            "causes": ["Viral infection (common cold, flu)", "Bacterial infection (strep, UTI, typhoid)", "Inflammatory reaction", "Heat exhaustion"],
            "self_care": [
                "Stay well hydrated with clean water, ORS (oral rehydration salts), or clear soups.",
                "Get plenty of rest so your immune system can fight the infection.",
                "Use a cool damp cloth on the forehead or take a lukewarm sponge bath.",
                "Wear light, comfortable cotton clothing and keep the room well-ventilated.",
            ],
            "otc_options": [
                "Paracetamol (Acetaminophen) 500mg-650mg every 4 to 6 hours as needed (Maximum 4000mg/day).",
                "Ibuprofen 400mg with food every 6 to 8 hours (if no stomach ulcer or kidney issues).",
            ],
            "red_flags": "Fever above 103°F (39.4°C), stiff neck, severe headache, confusion, skin rash, or fever lasting longer than 3 days.",
        },
        "headache": {
            "name": "Headache (Cephalea / Sar Dard)",
            "aliases": ["headache", "sar dard", "sir dard", "head pain", "tension headache"],
            "causes": ["Stress or eye strain", "Dehydration / lack of water", "Lack of sleep / fatigue", "Sinus congestion", "Caffeine withdrawal"],
            "self_care": [
                "Rest in a quiet, dark, well-ventilated room.",
                "Drink 500ml - 1 Liter of water immediately to rule out dehydration.",
                "Apply a cold or warm compress to your forehead or the back of your neck.",
                "Take a 20-minute break from phone and computer screens.",
            ],
            "otc_options": ["Paracetamol (500mg)", "Ibuprofen (400mg with meal)"],
            "red_flags": "Sudden 'thunderclap' headache (worst headache of your life), accompanied by fever, neck stiffness, confusion, or weakness on one side of your body.",
        },
        "cough": {
            "name": "Cough (Khansi / Respiratory Tract)",
            "aliases": ["cough", "khansi", "coughing", "dry cough", "wet cough", "phlegm"],
            "causes": ["Upper respiratory viral infection (cold/flu)", "Allergic irritation or post-nasal drip", "Acid reflux (GERD)", "Bronchitis or mild asthma"],
            "self_care": [
                "Drink warm water with honey and ginger (avoid honey for infants under 1 year).",
                "Do warm steam inhalation 2 times daily.",
                "Gargle with warm salt water (1/2 tsp salt in warm water) morning and night.",
            ],
            "otc_options": [
                "Dextromethorphan syrup for dry, hacking cough",
                "Guaifenesin or Ambroxol syrup for productive/phlegm cough",
                "Cetirizine (10mg) if cough is allergy-driven",
            ],
            "red_flags": "Coughing up blood, shortness of breath, high persistent fever, or cough lasting > 3 weeks.",
        },
        "sore throat": {
            "name": "Sore Throat (Pharyngitis / Gale me Dard)",
            "aliases": ["sore throat", "throat pain", "gale me dard", "gala kharab", "tonsils"],
            "causes": ["Viral pharyngitis (90% of cases)", "Bacterial streptococcal infection", "Dry room air", "Acid reflux"],
            "self_care": [
                "Gargle with warm salt water 3-4 times a day.",
                "Sip warm herbal teas, ginger tea, or warm water with lemon and honey.",
                "Suck on throat lozenges or hard candies to keep the throat moist.",
            ],
            "otc_options": ["Paracetamol or Ibuprofen for pain relief", "Antiseptic/anesthetic throat lozenges"],
            "red_flags": "Inability to swallow saliva, difficulty opening mouth (trismus), breathing difficulty, or high fever with white patches on tonsils.",
        },
        "stomach pain": {
            "name": "Abdominal Discomfort / Pet Dard",
            "aliases": ["stomach pain", "pet dard", "abdominal pain", "gastric", "indigestion", "cramps"],
            "causes": ["Gas, acidity, and bloating", "Indigestion (dyspepsia)", "Food poisoning / gastroenteritis", "Constipation"],
            "self_care": [
                "Follow a light, bland diet (BRAT: Bananas, Rice, Applesauce, Toast, or Khichdi).",
                "Avoid oily, spicy, acidic, fried foods, and carbonated sodas.",
                "Sip warm peppermint tea or ajwain (carom seed) warm water.",
                "Use a warm water heating bag on your lower stomach for 15 minutes.",
            ],
            "otc_options": [
                "Antacids (Gelusil, Digene, or Calcium Carbonate) for acidity and heartburn",
                "Simethicone for gas and bloating",
                "ORS solution if vomiting or loose motions are present",
            ],
            "red_flags": "Sharp severe pain in the lower right side (possible appendicitis), vomiting blood, black stools, high fever, or rigid tight abdomen.",
        },
        "migraine": {
            "name": "Migraine (Adhasisi / Throbbing Headache)",
            "aliases": ["migraine", "half headache", "throbbing head", "aura"],
            "causes": ["Neurovascular sensitivity triggered by light, noise, hormonal fluctuations, skipped meals, or stress"],
            "self_care": [
                "Lie down in a completely dark and silent bedroom.",
                "Place an ice pack or cold gel wrap across your forehead or temples.",
                "Stay well hydrated with electrolyte water.",
                "Practice slow 4-7-8 deep diaphragmatic breathing.",
            ],
            "otc_options": ["Ibuprofen (400mg) or Naproxen taken at the very first sign of headache", "Paracetamol + Caffeine"],
            "red_flags": "Visual loss, speech impairment, limb weakness, or fever.",
        },
        "blood pressure": {
            "name": "Hypertension & Blood Pressure Care",
            "aliases": ["blood pressure", "high bp", "hypertension", "bp", "low bp"],
            "causes": ["High dietary sodium, chronic stress, lack of exercise, obesity, genetics"],
            "self_care": [
                "Adopt the DASH diet: rich in green vegetables, fruits, whole grains, and low-fat dairy.",
                "Cut down dietary salt to less than 1 level teaspoon (2,300mg sodium) per day.",
                "Engage in 30 minutes of moderate brisk walking 5 days a week.",
                "Practice daily stress management and limit caffeine/alcohol.",
            ],
            "otc_options": ["Blood pressure medication must strictly be prescribed by a physician. Never alter doses on your own."],
            "red_flags": "Systolic > 180 mmHg or Diastolic > 120 mmHg with chest pain, breathlessness, or blurred vision (Emergency).",
        },
        "diabetes": {
            "name": "Diabetes & Blood Sugar Management",
            "aliases": ["diabetes", "sugar", "blood sugar", "glucose", "high sugar"],
            "causes": ["Insulin resistance (Type 2) or lack of insulin production (Type 1)"],
            "self_care": [
                "Eat high-fiber complex carbohydrates (oats, brown rice, legumes) and avoid refined sugars.",
                "Take a light 15-minute walk after lunch and dinner to help glucose absorption.",
                "Monitor your blood glucose at consistent times.",
                "Stay well hydrated throughout the day.",
            ],
            "otc_options": ["Prescribed insulin or oral hypoglycemics (Metformin). Keep fruit juice or glucose handy for hypoglycemia (< 70 mg/dL)."],
            "red_flags": "Fruity breath odor, extreme confusion, continuous vomiting, or blood sugar > 300 mg/dL.",
        },
        "stress": {
            "name": "Stress, Anxiety & Mental Wellbeing",
            "aliases": ["stress", "anxiety", "depression", "tension", "mental health", "panic", "worry"],
            "causes": ["Work pressure, academic exams, life transitions, lack of sleep, burnout"],
            "self_care": [
                "Practice box breathing: inhale for 4 seconds, hold for 4, exhale for 4, hold for 4.",
                "Go for a 20-minute walk outside in natural sunlight.",
                "Reduce screen time and avoid social media 1 hour before sleep.",
                "Share your feelings with a trusted friend, family member, or counselor.",
            ],
            "otc_options": ["Chamomile tea, magnesium supplements. For persistent anxiety, consult a licensed psychologist or psychiatrist."],
            "red_flags": "Feelings of hopelessness, severe panic attacks, or thoughts of self-harm (National Suicide Prevention Helpline: 988 or local crisis lines).",
        },
    }

    MEDICATIONS_DATABASE: Dict[str, Dict[str, Any]] = {
        "paracetamol": {
            "aliases": ["paracetamol", "acetaminophen", "crocin", "calpol", "tylenol", "dolo", "pcm"],
            "generic": "Acetaminophen / Paracetamol",
            "class": "Analgesic (Pain Reliever) & Antipyretic (Fever Reducer)",
            "indication": "Mild to moderate pain (headache, toothache, body ache) and fever reduction.",
            "usual_dose": "Adults: 500mg to 650mg every 4 to 6 hours as needed. Maximum daily dose is 4000mg (4 grams).",
            "precautions": "Do not take with other cold/cough medications that already contain paracetamol to avoid overdose. Avoid heavy alcohol intake to protect liver health.",
            "interactions": "Safe with most drugs; caution with long-term high-dose Warfarin.",
        },
        "ibuprofen": {
            "aliases": ["ibuprofen", "advil", "motrin", "brufen", "combiflam"],
            "generic": "Ibuprofen",
            "class": "Non-Steroidal Anti-Inflammatory Drug (NSAID)",
            "indication": "Inflammatory pain, swelling, menstrual cramps, toothache, arthritis, fever.",
            "usual_dose": "Adults: 200mg to 400mg every 6 to 8 hours ALWAYS taken with food or milk.",
            "precautions": "Always take with food to protect your stomach lining. Avoid if you have active stomach ulcers, severe kidney disease, or in late pregnancy.",
            "interactions": "Do not combine with other NSAIDs (Aspirin, Naproxen) or blood thinners without a doctor's guidance.",
        },
        "cetirizine": {
            "aliases": ["cetirizine", "zyrtec", "cetzine", "allegra", "antihistamine"],
            "generic": "Cetirizine Hydrochloride",
            "class": "Second-generation Antihistamine",
            "indication": "Allergies, runny nose, sneezing, itchy eyes, hives, skin rash.",
            "usual_dose": "Adults: 10mg once daily (preferably in the evening).",
            "precautions": "May cause mild drowsiness in some individuals. Avoid alcohol and operating heavy machinery.",
            "interactions": "Sedatives and sleeping pills may increase drowsiness.",
        },
        "omeprazole": {
            "aliases": ["omeprazole", "prilosec", "omez", "pantoprazole", "pantocid", "antacid", "acidity"],
            "generic": "Omeprazole (PPI)",
            "class": "Proton Pump Inhibitor (Acid Reducer)",
            "indication": "Acid reflux, GERD, heartburn, stomach ulcers, gastritis.",
            "usual_dose": "20mg once daily, taken 30-60 minutes before breakfast on an empty stomach.",
            "precautions": "Meant for short-term courses (14 days OTC). If acidity persists, consult a physician.",
            "interactions": "May reduce absorption of iron and vitamin B12.",
        },
    }

    @classmethod
    def check_emergency(cls, text: str) -> Optional[str]:
        """Detect critical emergency keywords in user input."""
        lower = text.lower()
        for red_flag in cls.EMERGENCY_RED_FLAGS:
            if red_flag in lower:
                return red_flag
        return None

    @classmethod
    def match_symptom(cls, text: str) -> Optional[Dict[str, Any]]:
        """Match common symptom from knowledge base via keywords/aliases."""
        lower = text.lower()
        for key, data in cls.SYMPTOM_DATABASE.items():
            for alias in data.get("aliases", [key]):
                if alias in lower:
                    return data
        return None

    @classmethod
    def match_medication(cls, text: str) -> Optional[Dict[str, Any]]:
        """Match common medication query via generic or brand names."""
        lower = text.lower()
        for key, data in cls.MEDICATIONS_DATABASE.items():
            for alias in data.get("aliases", [key]):
                if alias in lower:
                    return data
        return None

    @classmethod
    def generate_expert_response(cls, query: str) -> str:
        """
        Generate structured, clinical-grade medical response using embedded knowledge base.
        Zero external API key needed.
        """
        lower = query.lower().strip()

        # 0. Greetings / Hello
        if lower in ["hello", "hi", "hey", "namaste", "halo", "good morning", "good evening", "good afternoon"]:
            return (
                "👋 **Hello! I am HealthPulse AI, your Healthcare Virtual Assistant.**\n\n"
                "I am here to help you with:\n"
                "- 🩺 **Symptoms & Doubts:** Fever, headaches, cough, stomach pain, migraines, etc.\n"
                "- 💊 **Medications & Precautions:** Safe dosages, interactions (e.g., Paracetamol, Ibuprofen).\n"
                "- 📋 **Clinical Symptom Triage:** Assessing severity and whether you need self-care or a doctor.\n"
                "- 💓 **Vital Signs Check:** Blood pressure, heart rate, blood sugar, and oxygen levels.\n\n"
                "How can I assist you with your health today? Please describe what you are feeling!"
            )

        # 1. Check for critical medical emergencies
        flag = cls.check_emergency(query)
        if flag:
            return (
                f"🚨 **CRITICAL MEDICAL ALERT: IMMEDIATE ATTENTION REQUIRED**\n\n"
                f"You mentioned symptoms related to **'{flag}'**, which can be a sign of a life-threatening medical emergency.\n\n"
                f"### Immediate Actions:\n"
                f"1. **Call Emergency Services Immediately:** Dial **911** (US), **112** (India/Europe), or your local emergency medical number.\n"
                f"2. **Do not drive yourself:** Request an ambulance so paramedics can initiate emergency care on the way.\n"
                f"3. **If chest pain / heart attack is suspected:** Sit down, stay calm, loosen tight clothing, and if you have no allergies and medical approval, chew one adult aspirin (325mg).\n"
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
                f"*Clinical Note: If symptoms persist beyond 48-72 hours, worsen significantly, or do not respond to self-care, schedule an in-person medical consultation.*"
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
            f"💬 *Feel free to describe specific symptoms (e.g., fever, headache, blood pressure reading) or use the Symptom Triage tab above for a structured evaluation!*"
        )
