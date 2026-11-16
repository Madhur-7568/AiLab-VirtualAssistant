"""
HMM Intent and Slot Tagger for Virtual Assistant
------------------------------------------------
Uses a Hidden Markov Model with Viterbi decoding to tokenize and tag
natural language user utterances into semantic dialogue states.
"""

import re
from typing import List, Dict, Any, Tuple
import numpy as np
from src.models.hidden_markov_model import HiddenMarkovModel


class HMMIntentTagger:
    """
    Sequence tagger mapping user utterance tokens to semantic dialogue states:
    - GREETING
    - INTENT_SCHEDULE
    - INTENT_REMINDER
    - INTENT_QUERY
    - INTENT_HEALTH
    - SLOT_TIME
    - SLOT_URGENCY
    - SLOT_TASK
    - OTHER
    """

    STATES = [
        "GREETING",
        "INTENT_SCHEDULE",
        "INTENT_REMINDER",
        "INTENT_QUERY",
        "INTENT_HEALTH",
        "SLOT_TIME",
        "SLOT_URGENCY",
        "SLOT_TASK",
        "OTHER",
    ]

    VOCABULARY = [
        # Greetings
        "hello", "hi", "hey", "morning", "afternoon", "evening", "greetings",
        # Schedule tokens
        "schedule", "book", "set", "plan", "calendar", "appoint", "appointment",
        # Reminder tokens
        "remind", "reminder", "alarm", "notify", "alert", "ping",
        # Query tokens
        "what", "when", "how", "where", "why", "who", "check", "tell", "weather", "traffic", "status",
        # Health tokens
        "health", "doctor", "medicine", "pill", "vitals", "heartrate", "fever", "pain", "hospital",
        # Time tokens
        "today", "tomorrow", "tonight", "morning", "afternoon", "evening", "night",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "at", "pm", "am", "now", "soon", "later", "9am", "10am", "2pm", "5pm",
        # Urgency tokens
        "urgent", "urgently", "emergency", "asap", "immediately", "critical", "important", "quick", "fast",
        # Task tokens
        "meeting", "lab", "assignment", "project", "exam", "report", "call", "presentation", "interview", "class",
        # Function / Other tokens
        "me", "a", "an", "the", "for", "to", "in", "on", "with", "my", "is", "it", "please", "can", "you",
        "<UNK>",
    ]

    def __init__(self):
        self.states = list(self.STATES)
        # Deduplicate vocabulary while preserving order
        self.vocab = list(dict.fromkeys(self.VOCABULARY))
        self.hmm = self._build_default_hmm()

    def _build_default_hmm(self) -> HiddenMarkovModel:
        num_states = len(self.states)
        num_obs = len(self.vocab)

        # Initial distribution pi: conversational utterances usually start with GREETING, INTENT, or OTHER
        pi = np.full(num_states, 0.05)
        pi[self.states.index("GREETING")] = 0.35
        pi[self.states.index("INTENT_SCHEDULE")] = 0.15
        pi[self.states.index("INTENT_REMINDER")] = 0.15
        pi[self.states.index("INTENT_QUERY")] = 0.15
        pi[self.states.index("INTENT_HEALTH")] = 0.05
        pi[self.states.index("OTHER")] = 0.10
        pi = pi / np.sum(pi)

        # Transition matrix A: natural dialogue flow (e.g. GREETING -> INTENT -> TASK/URGENCY -> TIME)
        A = np.full((num_states, num_states), 0.05)
        s_idx = {s: i for i, s in enumerate(self.states)}

        # From GREETING
        A[s_idx["GREETING"], s_idx["GREETING"]] = 0.05
        A[s_idx["GREETING"], s_idx["INTENT_SCHEDULE"]] = 0.25
        A[s_idx["GREETING"], s_idx["INTENT_REMINDER"]] = 0.20
        A[s_idx["GREETING"], s_idx["INTENT_QUERY"]] = 0.20
        A[s_idx["GREETING"], s_idx["INTENT_HEALTH"]] = 0.10
        A[s_idx["GREETING"], s_idx["OTHER"]] = 0.20

        # From INTENTS -> SLOTS
        for intent in ["INTENT_SCHEDULE", "INTENT_REMINDER", "INTENT_QUERY", "INTENT_HEALTH"]:
            A[s_idx[intent], s_idx["SLOT_TASK"]] = 0.35
            A[s_idx[intent], s_idx["SLOT_URGENCY"]] = 0.25
            A[s_idx[intent], s_idx["SLOT_TIME"]] = 0.20
            A[s_idx[intent], s_idx["OTHER"]] = 0.20

        # From SLOTS -> OTHER SLOTS
        A[s_idx["SLOT_TASK"], s_idx["SLOT_TIME"]] = 0.35
        A[s_idx["SLOT_TASK"], s_idx["SLOT_URGENCY"]] = 0.25
        A[s_idx["SLOT_TASK"], s_idx["OTHER"]] = 0.25

        A[s_idx["SLOT_URGENCY"], s_idx["SLOT_TASK"]] = 0.35
        A[s_idx["SLOT_URGENCY"], s_idx["SLOT_TIME"]] = 0.30
        A[s_idx["SLOT_URGENCY"], s_idx["OTHER"]] = 0.20

        A[s_idx["SLOT_TIME"], s_idx["SLOT_URGENCY"]] = 0.25
        A[s_idx["SLOT_TIME"], s_idx["SLOT_TASK"]] = 0.25
        A[s_idx["SLOT_TIME"], s_idx["OTHER"]] = 0.30

        # Normalize rows of A
        A = A / np.sum(A, axis=1, keepdims=True)

        # Emission matrix B
        B = np.full((num_states, num_obs), 0.001)

        # Helper to boost emissions for state
        def boost_emissions(state_name: str, words: List[str], weight: float = 1.0):
            idx = s_idx[state_name]
            for w in words:
                if w in self.vocab:
                    B[idx, self.vocab.index(w)] += weight

        boost_emissions("GREETING", ["hello", "hi", "hey", "greetings", "morning", "afternoon", "evening"], 10.0)
        boost_emissions("INTENT_SCHEDULE", ["schedule", "book", "set", "plan", "calendar", "appoint", "appointment"], 10.0)
        boost_emissions("INTENT_REMINDER", ["remind", "reminder", "alarm", "notify", "alert", "ping"], 10.0)
        boost_emissions("INTENT_QUERY", ["what", "when", "how", "where", "why", "who", "check", "tell", "weather", "traffic", "status"], 10.0)
        boost_emissions("INTENT_HEALTH", ["health", "doctor", "medicine", "pill", "vitals", "heartrate", "fever", "pain", "hospital"], 10.0)
        boost_emissions("SLOT_TIME", ["today", "tomorrow", "tonight", "morning", "afternoon", "evening", "night", "monday", "tuesday", "wednesday", "thursday", "friday", "at", "pm", "am", "now", "soon", "later", "9am", "10am", "2pm", "5pm"], 10.0)
        boost_emissions("SLOT_URGENCY", ["urgent", "urgently", "emergency", "asap", "immediately", "critical", "important", "quick", "fast"], 10.0)
        boost_emissions("SLOT_TASK", ["meeting", "lab", "assignment", "project", "exam", "report", "call", "presentation", "interview", "class"], 10.0)
        boost_emissions("OTHER", ["me", "a", "an", "the", "for", "to", "in", "on", "with", "my", "is", "it", "please", "can", "you", "<UNK>"], 8.0)

        # Normalize rows of B
        B = B / np.sum(B, axis=1, keepdims=True)

        return HiddenMarkovModel(
            states=self.states,
            observations=self.vocab,
            start_prob=pi,
            trans_prob=A,
            emission_prob=B,
        )

    def tokenize(self, text: str) -> List[str]:
        """Simple clean tokenization of user utterance."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [t.strip() for t in cleaned.split() if t.strip()]
        return tokens if tokens else ["<UNK>"]

    def tag_utterance(self, text: str) -> Dict[str, Any]:
        """
        Tag utterance tokens using Viterbi decoding and extract semantics.
        """
        tokens = self.tokenize(text)
        obs_indices = self.hmm.encode_observations(tokens)

        # Run Viterbi algorithm
        decoded_states, log_prob, trellis = self.hmm.viterbi(obs_indices)

        # Also compute posterior probabilities gamma_t(i)
        gamma = self.hmm.posterior_decoding(obs_indices)

        tagged_tokens: List[Dict[str, Any]] = []
        for i, (tok, st) in enumerate(zip(tokens, decoded_states)):
            tagged_tokens.append({
                "token": tok,
                "state": st,
                "confidence": float(np.max(gamma[i, :])),
                "posteriors": {
                    self.states[k]: float(gamma[i, k]) for k in range(len(self.states))
                },
            })

        # Infer overall primary intent
        intent_candidates = [
            st for st in decoded_states
            if st in ["INTENT_SCHEDULE", "INTENT_REMINDER", "INTENT_QUERY", "INTENT_HEALTH", "GREETING"]
        ]
        primary_intent = intent_candidates[0] if intent_candidates else "INTENT_QUERY"

        # Extract slots
        urgency_slot = any(st == "SLOT_URGENCY" for st in decoded_states)
        time_tokens = [tok for tok, st in zip(tokens, decoded_states) if st == "SLOT_TIME"]
        task_tokens = [tok for tok, st in zip(tokens, decoded_states) if st == "SLOT_TASK"]

        return {
            "text": text,
            "tokens": tokens,
            "decoded_states": decoded_states,
            "primary_intent": primary_intent,
            "is_urgent": urgency_slot,
            "extracted_slots": {
                "time": " ".join(time_tokens) if time_tokens else None,
                "task": " ".join(task_tokens) if task_tokens else None,
                "urgency": "High" if urgency_slot else "Normal",
            },
            "viterbi_log_prob": log_prob,
            "tagged_tokens": tagged_tokens,
        }
