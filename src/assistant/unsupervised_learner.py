"""
Unsupervised Conversational Learner via EM (Baum-Welch)
-------------------------------------------------------
Uses the Expectation-Maximization algorithm to learn and refine
dialogue state transition dynamics and token emissions from raw, unannotated
conversational text corpora.
"""

from typing import List, Dict, Any
from src.models.em_algorithm import BaumWelchLearner
from src.assistant.intent_tagger import HMMIntentTagger


class UnsupervisedDialogueLearner:
    """
    Trains and refines an HMMIntentTagger using Baum-Welch EM.
    """

    SAMPLE_CONVERSATIONAL_CORPUS = [
        "hello virtual assistant",
        "hi good morning can you help me",
        "schedule urgent lab meeting tomorrow morning",
        "book a calendar appointment for friday at 2pm",
        "set an alarm for 9am tomorrow",
        "remind me to complete my ai lab assignment immediately",
        "what is the weather today",
        "tell me the status of my project",
        "how can i solve this problem",
        "emergency call the doctor right now",
        "check my health vitals urgently",
        "hey schedule a quick call with teacher",
        "notify me when the class begins",
        "good afternoon please check traffic",
        "i need immediate medicine assistance",
    ]

    def __init__(self, tagger: HMMIntentTagger):
        self.tagger = tagger
        self.learner = BaumWelchLearner(max_iter=30, tol=1e-3, pseudocount=1e-2)
        self.training_history: List[float] = []

    def train_on_corpus(
        self,
        corpus: List[str] = None,
        iterations: int = 25,
    ) -> Dict[str, Any]:
        """
        Run Baum-Welch EM on the corpus.

        Returns:
            Dictionary with training stats (initial vs final log-likelihood, iterations).
        """
        if corpus is None:
            corpus = self.SAMPLE_CONVERSATIONAL_CORPUS

        self.learner.max_iter = iterations

        # Convert text corpus to encoded integer sequences
        encoded_seqs = [
            self.tagger.hmm.encode_observations(self.tagger.tokenize(sentence))
            for sentence in corpus
        ]

        # Calculate initial log-likelihood
        initial_log_lik = sum(self.tagger.hmm.forward(seq)[2] for seq in encoded_seqs)

        # Run Baum-Welch EM
        self.learner.fit(self.tagger.hmm, encoded_seqs)
        self.training_history = list(self.learner.log_likelihood_history)

        final_log_lik = (
            self.training_history[-1] if self.training_history else initial_log_lik
        )

        return {
            "num_utterances": len(corpus),
            "iterations_run": len(self.training_history),
            "initial_log_likelihood": float(initial_log_lik),
            "final_log_likelihood": float(final_log_lik),
            "log_likelihood_curve": [float(x) for x in self.training_history],
            "converged": len(self.training_history) < iterations,
        }
