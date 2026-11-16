"""
Interactive Command-Line Interface (CLI)
Probabilistic Virtual Assistant
"""

import sys
from src.assistant.pipeline import ProbabilisticVirtualAssistant


def print_banner():
    banner = """
========================================================================
   PROBABILISTIC VIRTUAL ASSISTANT (HMM + EM + BAYESIAN NETWORK)
   Lab Assignment Project - Madhur Dhillon
========================================================================
Commands:
  :train    -> Run Baum-Welch EM training on conversational dialogue corpus
  :inspect  -> Toggle detailed probabilistic inference inspection mode
  :quit     -> Exit Virtual Assistant
------------------------------------------------------------------------
"""
    print(banner)


def main():
    print_banner()
    assistant = ProbabilisticVirtualAssistant()
    inspect_mode = True

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            if user_input.lower() in [":quit", ":exit", "quit", "exit"]:
                print("\nGoodbye! Thank you for using Probabilistic Virtual Assistant.")
                break

            elif user_input.lower() == ":inspect":
                inspect_mode = not inspect_mode
                status = "ON" if inspect_mode else "OFF"
                print(f"[System] Detailed inspection mode is now {status}.")
                continue

            elif user_input.lower() == ":train":
                print("\n[EM Learner] Running Baum-Welch Expectation-Maximization...")
                stats = assistant.train_unsupervised_em(iterations=15)
                print(f"  Utterances processed: {stats['num_utterances']}")
                print(f"  Initial Log-Likelihood: {stats['initial_log_likelihood']:.4f}")
                print(f"  Final Log-Likelihood:   {stats['final_log_likelihood']:.4f}")
                print(f"  Converged: {stats['converged']}")
                print("[EM Learner] Parameters successfully updated.")
                continue

            # Process normal utterance
            result = assistant.process_utterance(user_input)

            # Assistant response
            print(f"\nAssistant > {result['response']}")

            if inspect_mode:
                print("\n" + "-" * 50)
                print(" [PROBABILISTIC TRACE & MATHEMATICAL REASONING]")
                print("-" * 50)

                # 1. HMM Trace
                print("1. Hidden Markov Model (HMM) Viterbi Decoding:")
                print(f"   Decoded Path: {' -> '.join(result['hmm']['decoded_states'])}")
                print(f"   Primary Intent: {result['hmm']['primary_intent']}")
                print(f"   Slots Extracted: {result['hmm']['slots']}")
                print(f"   Viterbi Log-Prob: {result['hmm']['viterbi_log_prob']:.4f}")

                # 2. Bayesian Network Trace
                print("\n2. Bayesian Network Variable Elimination:")
                print(f"   Conditioned Evidence: {result['bayesian_network']['evidence']}")
                print(f"   Selected Action: {result['bayesian_network']['selected_action']} (Confidence: {result['bayesian_network']['confidence'] * 100:.2f}%)")
                print("   Action Posterior Distribution:")
                for act, prob in result['bayesian_network']['action_probabilities'].items():
                    bar = "#" * int(prob * 30)
                    print(f"     {act:<26} [{bar:<30}] {prob * 100:5.1f}%")
                print("-" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
