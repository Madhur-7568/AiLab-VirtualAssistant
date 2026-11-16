"""
Interactive Command-Line Healthcare Assistant (CLI)
HealthPulse AI: Clinical Assistant & Doubt Solver
"""

import sys
from src.assistant.health_assistant import HealthAssistant


def print_banner():
    banner = """
========================================================================
   🏥 HEALTHPULSE AI - HEALTHCARE VIRTUAL ASSISTANT & MEDICAL CHATBOT
   Author: Madhur Dhillon
========================================================================
Commands:
  :triage   -> Launch interactive clinical symptom checker
  :vitals   -> Launch vital signs risk analyzer
  :key      -> Set or update Google Gemini API key
  :quit     -> Exit Virtual Assistant
------------------------------------------------------------------------
Ask any medical doubt, symptom question, or medicine inquiry below:
"""
    print(banner)


def run_interactive_triage(assistant: HealthAssistant):
    print("\n--- CLINICAL SYMPTOM TRIAGE ---")
    symptom = input("What is your primary symptom? ").strip()
    if not symptom:
        return
    try:
        sev_str = input("Rate severity from 1 (mild) to 10 (unbearable) [default 4]: ").strip()
        severity = int(sev_str) if sev_str else 4
        dur_str = input("How many days have you had this symptom? [default 1]: ").strip()
        duration = int(dur_str) if dur_str else 1
        chronic = input("Do you have any chronic conditions (diabetes, heart disease)? (y/N): ").strip().lower() == "y"

        res = assistant.check_symptoms(symptom, severity, duration, chronic=chronic)
        print(f"\n[TRIAGE RESULT] {res['title']}")
        print(f"Action:    {res['recommended_action']}")
        print(f"Timeframe: {res['timeframe']}")
        print(f"Rationale: {res['clinical_rationale']}")
    except ValueError:
        print("Invalid input.")


def run_interactive_vitals(assistant: HealthAssistant):
    print("\n--- VITAL SIGNS EVALUATOR ---")
    try:
        bp = input("Enter Blood Pressure as Systolic/Diastolic (e.g. 120/80) or press Enter to skip: ").strip()
        sys_val, dia_val = None, None
        if "/" in bp:
            parts = bp.split("/")
            sys_val, dia_val = int(parts[0]), int(parts[1])

        hr_str = input("Enter Resting Heart Rate (BPM) or press Enter to skip: ").strip()
        hr = int(hr_str) if hr_str else None

        res = assistant.analyze_vitals(systolic=sys_val, diastolic=dia_val, heart_rate=hr)
        print("\n[VITALS ANALYSIS]")
        for k, v in res.items():
            print(f"  • {k.replace('_', ' ').title()}: {v['reading']} -> {v['category']} ({v['clinical_advice']})")
    except Exception as e:
        print(f"Error parsing vitals: {e}")


def main():
    print_banner()
    assistant = HealthAssistant()
    user_api_key = None

    while True:
        try:
            query = input("\nYou > ").strip()
            if not query:
                continue

            if query.lower() in [":quit", ":exit", "quit", "exit"]:
                print("\nTake care of your health! Goodbye.")
                break

            elif query.lower() == ":triage":
                run_interactive_triage(assistant)
                continue

            elif query.lower() == ":vitals":
                run_interactive_vitals(assistant)
                continue

            elif query.lower() == ":key":
                key_input = input("Enter your Gemini API key (or leave empty to reset): ").strip()
                user_api_key = key_input if key_input else None
                status = "Set" if user_api_key else "Cleared (using clinical knowledge fallback)"
                print(f"[System] API Key {status}.")
                continue

            # Process conversational healthcare query
            print("\nHealthPulse AI is thinking...")
            result = assistant.chat(query, api_key_override=user_api_key)

            print(f"\nAssistant [{result['model_used']}] >\n")
            print(result["response"])
            print("\n" + "-" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Stay healthy!")
            sys.exit(0)


if __name__ == "__main__":
    main()
