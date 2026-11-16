"""
Virtual Assistant Modules
Integrates HMM sequence tagging, EM unsupervised parameter estimation,
and Bayesian Network decision making into an interactive conversational assistant.
"""

from src.assistant.intent_tagger import HMMIntentTagger
from src.assistant.decision_engine import BayesianDecisionEngine
from src.assistant.pipeline import ProbabilisticVirtualAssistant

__all__ = [
    "HMMIntentTagger",
    "BayesianDecisionEngine",
    "ProbabilisticVirtualAssistant",
]
