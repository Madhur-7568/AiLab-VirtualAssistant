"""
Unified Probabilistic Virtual Assistant Pipeline
-----------------------------------------------
Integrates:
1. Hidden Markov Model (HMM) sequence decoding with Viterbi
2. Expectation-Maximization (EM / Baum-Welch) unsupervised model learning
3. Bayesian Network (BN) variable elimination for context-aware decision making
"""

from typing import Dict, Any, Optional
import datetime
from src.assistant.intent_tagger import HMMIntentTagger
from src.assistant.decision_engine import BayesianDecisionEngine
from src.assistant.unsupervised_learner import UnsupervisedDialogueLearner


class ProbabilisticVirtualAssistant:
    """
    End-to-end Virtual Assistant that orchestrates HMM, EM, and Bayesian Networks.
    """

    def __init__(self):
        self.tagger = HMMIntentTagger()
        self.decision_engine = BayesianDecisionEngine()
        self.learner = UnsupervisedDialogueLearner(self.tagger)
        self.dialogue_history = []

    def get_current_time_of_day(self) -> str:
        """Infer current time category."""
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"

    def process_utterance(
        self,
        utterance: str,
        user_sentiment: str = "Calm",
        time_of_day: Optional[str] = None,
        override_urgency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user input through the probabilistic pipeline:
        1. HMM Viterbi decoding of utterance tokens
        2. Bayesian Network variable elimination inference
        3. Response generation and probabilistic trace
        """
        if not time_of_day:
            time_of_day = self.get_current_time_of_day()

        # Step 1: HMM Sequence Tagging
        hmm_result = self.tagger.tag_utterance(utterance)

        # Determine urgency evidence
        if override_urgency:
            urgency_evidence = override_urgency
        elif hmm_result["is_urgent"]:
            urgency_evidence = "High"
        elif hmm_result["primary_intent"] == "INTENT_HEALTH":
            urgency_evidence = "Critical"
        else:
            urgency_evidence = "Normal"

        # Step 2: Bayesian Decision Inference
        bn_result = self.decision_engine.infer_action(
            detected_intent=hmm_result["primary_intent"],
            user_urgency=urgency_evidence,
            time_of_day=time_of_day,
            user_sentiment=user_sentiment,
        )

        # Step 3: Response Generation
        response_text = self._generate_response(hmm_result, bn_result)

        turn_result = {
            "user_input": utterance,
            "response": response_text,
            "hmm": {
                "tokens": hmm_result["tokens"],
                "decoded_states": hmm_result["decoded_states"],
                "primary_intent": hmm_result["primary_intent"],
                "slots": hmm_result["extracted_slots"],
                "viterbi_log_prob": hmm_result["viterbi_log_prob"],
                "tagged_tokens": hmm_result["tagged_tokens"],
            },
            "bayesian_network": {
                "evidence": bn_result["evidence"],
                "selected_action": bn_result["best_action"],
                "confidence": bn_result["best_action_confidence"],
                "action_probabilities": bn_result["action_posteriors"],
                "importance_probabilities": bn_result["importance_posteriors"],
                "fatigue_probabilities": bn_result["fatigue_posteriors"],
                "require_confirmation": bn_result["require_confirmation"],
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }

        self.dialogue_history.append(turn_result)
        return turn_result

    def _generate_response(self, hmm_res: Dict[str, Any], bn_res: Dict[str, Any]) -> str:
        """Construct natural language response based on inferred action and extracted slots."""
        action = bn_res["best_action"]
        slots = hmm_res["extracted_slots"]
        task = slots.get("task") or "your task"
        time_slot = slots.get("time") or "the scheduled time"

        if action == "PoliteGreeting":
            return "Hello! I am your Probabilistic Virtual Assistant. How can I assist you with your schedule, tasks, or queries today?"

        elif action == "BookCalendar":
            return f"I have prepared a calendar booking for '{task}' at {time_slot}. (Bayesian confidence: {bn_res['best_action_confidence'] * 100:.1f}%)"

        elif action == "SetImmediateAlarm":
            return f"High priority reminder set for '{task}' ({time_slot}). I will alert you proactively!"

        elif action == "ProvideConciseAnswer":
            return f"Here is the concise answer regarding '{task}': All systems and schedules are functioning normally."

        elif action == "TriggerEmergencyProtocol":
            return f"EMERGENCY PROTOCOL ACTIVATED: Contacting primary emergency services and dispatching critical notifications immediately."

        elif action == "AskClarification":
            return f"I recognized your request for '{task}', but could you please clarify the specific time and priority?"

        return "Request processed successfully via Bayesian decision inference."

    def train_unsupervised_em(self, corpus=None, iterations: int = 15) -> Dict[str, Any]:
        """Trigger Baum-Welch EM training on conversational utterances."""
        return self.learner.train_on_corpus(corpus=corpus, iterations=iterations)
