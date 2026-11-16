"""
Educational Demo Script: Probabilistic Virtual Assistant
--------------------------------------------------------
Demonstrates the three core lab assignment topics:
1. Hidden Markov Model (HMM) & Viterbi Decoding
2. Expectation-Maximization (EM / Baum-Welch) Parameter Learning
3. Bayesian Network & Exact Variable Elimination Inference
"""

import numpy as np
from src.models.hidden_markov_model import HiddenMarkovModel
from src.models.em_algorithm import BaumWelchLearner
from src.models.bayesian_network import BayesianNetwork
from src.assistant.pipeline import ProbabilisticVirtualAssistant


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def demo_hmm():
    print_section("Topic 1: Hidden Markov Model (HMM) & Viterbi Decoding")

    states = ["GREETING", "INTENT_SCHEDULE", "SLOT_TASK", "SLOT_TIME"]
    obs = ["hello", "schedule", "meeting", "tomorrow", "<UNK>"]

    pi = np.array([0.7, 0.2, 0.05, 0.05])
    A = np.array([
        [0.1, 0.6, 0.2, 0.1],
        [0.0, 0.1, 0.7, 0.2],
        [0.0, 0.0, 0.2, 0.8],
        [0.0, 0.0, 0.0, 1.0],
    ])
    # Emission B: shape (4 states, 5 obs)
    B = np.array([
        [0.80, 0.05, 0.05, 0.05, 0.05],  # GREETING
        [0.05, 0.80, 0.05, 0.05, 0.05],  # INTENT_SCHEDULE
        [0.05, 0.05, 0.80, 0.05, 0.05],  # SLOT_TASK
        [0.05, 0.05, 0.05, 0.80, 0.05],  # SLOT_TIME
    ])

    hmm = HiddenMarkovModel(states, obs, pi, A, B)
    sample_obs = ["hello", "schedule", "meeting", "tomorrow"]
    encoded = hmm.encode_observations(sample_obs)

    # 1. Forward Algorithm
    alpha, scale, log_lik = hmm.forward(encoded)
    print(f"\n1. Forward Evaluation:")
    print(f"   Observation Sequence: {sample_obs}")
    print(f"   Total Log-Likelihood log P(O | lambda): {log_lik:.4f}")

    # 2. Viterbi Decoding
    best_path, log_prob, _ = hmm.viterbi(encoded)
    print(f"\n2. Viterbi Optimal Hidden Path:")
    for word, state in zip(sample_obs, best_path):
        print(f"   '{word}' ---> [{state}]")
    print(f"   Path Log-Probability: {log_prob:.4f}")

    # 3. Posterior State Marginals
    gamma = hmm.posterior_decoding(encoded)
    print(f"\n3. Posterior State Probabilities gamma_t(i):")
    for t, word in enumerate(sample_obs):
        top_s_idx = np.argmax(gamma[t])
        print(f"   t={t} ('{word}'): State={states[top_s_idx]} (p={gamma[t, top_s_idx]:.4f})")


def demo_em():
    print_section("Topic 2: Expectation-Maximization (Baum-Welch Algorithm)")

    states = ["State_A", "State_B"]
    obs = ["token_1", "token_2"]
    # Initialize with perturbed parameters
    pi = np.array([0.5, 0.5])
    A = np.array([[0.55, 0.45], [0.45, 0.55]])
    B = np.array([[0.6, 0.4], [0.4, 0.6]])
    hmm = HiddenMarkovModel(states, obs, pi, A, B)

    # Unannotated observation sequences
    sequences = [
        [0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]

    learner = BaumWelchLearner(max_iter=10, tol=1e-3, verbose=True)
    print("Initial Transition Matrix A:\n", hmm.A)
    print("Initial Emission Matrix B:\n", hmm.B)

    print("\nRunning Baum-Welch EM iterations...")
    learner.fit(hmm, sequences)

    print("\nRe-estimated Transition Matrix A (after EM):\n", np.round(hmm.A, 4))
    print("Re-estimated Emission Matrix B (after EM):\n", np.round(hmm.B, 4))
    print(f"Log-Likelihood Improvement: {learner.log_likelihood_history[0]:.4f} -> {learner.log_likelihood_history[-1]:.4f}")


def demo_bayesian_network():
    print_section("Topic 3: Bayesian Network & Variable Elimination")

    bn = BayesianNetwork()
    bn.add_node("Urgency", ["Normal", "High"])
    bn.add_node("Intent", ["Schedule", "Emergency"])
    bn.add_node("Action", ["BookCalendar", "TriggerAlarm"])

    bn.add_edge("Urgency", "Action")
    bn.add_edge("Intent", "Action")

    bn.set_cpt("Urgency", np.array([0.8, 0.2]))
    bn.set_cpt("Intent", np.array([0.85, 0.15]))

    # P(Action | Urgency, Intent) -> shape (2, 2, 2)
    # Actions: [BookCalendar, TriggerAlarm]
    cpt_action = np.array([
        # Urgency=Normal
        [[0.90, 0.10],   # Intent=Schedule
         [0.20, 0.80]],  # Intent=Emergency
        # Urgency=High
        [[0.40, 0.60],   # Intent=Schedule
         [0.02, 0.98]],  # Intent=Emergency
    ])
    bn.set_cpt("Action", cpt_action)

    print("Network Topology: Urgency -> Action <- Intent")
    print("\nQuerying: P(Action | Urgency=High, Intent=Emergency)")
    posterior = bn.variable_elimination("Action", evidence={"Urgency": "High", "Intent": "Emergency"})
    for act, prob in posterior.items():
        print(f"   P({act}) = {prob * 100:.2f}%")


def demo_integrated_assistant():
    print_section("Topic 4: Integrated Virtual Assistant")

    assistant = ProbabilisticVirtualAssistant()

    test_queries = [
        "Hello good morning assistant",
        "Schedule urgent lab meeting tomorrow morning",
        "Emergency call the doctor right now",
    ]

    for q in test_queries:
        print(f"\n>>> USER: \"{q}\"")
        res = assistant.process_utterance(q)
        print(f"<<< ASSISTANT: {res['response']}")
        print(f"    [HMM Inferred Intent]: {res['hmm']['primary_intent']}")
        print(f"    [HMM Extracted Slots]: {res['hmm']['slots']}")
        print(f"    [Bayesian Action]:     {res['bayesian_network']['selected_action']} (Confidence: {res['bayesian_network']['confidence'] * 100:.1f}%)")


if __name__ == "__main__":
    demo_hmm()
    demo_em()
    demo_bayesian_network()
    demo_integrated_assistant()
    print("\n[SUCCESS] All probabilistic models and assistant demonstrations completed successfully!\n")
