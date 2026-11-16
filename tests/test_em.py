"""
Unit tests for Expectation-Maximization (EM) Algorithm
Tests Baum-Welch training of HMM parameters and log-likelihood monotonicity.
"""

import numpy as np
from src.models.hidden_markov_model import HiddenMarkovModel
from src.models.em_algorithm import BaumWelchLearner, GaussianMixtureEM


def test_baum_welch_convergence():
    states = ["S1", "S2"]
    observations = ["A", "B"]

    # Start with perturbed parameters
    pi = np.array([0.5, 0.5])
    A = np.array([[0.6, 0.4], [0.4, 0.6]])
    B = np.array([[0.7, 0.3], [0.3, 0.7]])
    hmm = HiddenMarkovModel(states, observations, pi, A, B)

    # Synthetic sequences generated with a pattern: S1 -> A, S2 -> B
    sequences = [
        [0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1],
    ]

    learner = BaumWelchLearner(max_iter=15, tol=1e-3, pseudocount=1e-2)
    trained_hmm = learner.fit(hmm, sequences)

    # 1. Row sums must remain stochastic
    assert np.isclose(np.sum(trained_hmm.pi), 1.0)
    assert np.allclose(np.sum(trained_hmm.A, axis=1), 1.0)
    assert np.allclose(np.sum(trained_hmm.B, axis=1), 1.0)

    # 2. History must exist and final log likelihood should be >= initial
    history = learner.log_likelihood_history
    assert len(history) > 1
    # EM guarantees non-decreasing or stabilized likelihood
    assert history[-1] >= history[0] - 1e-4


def test_gaussian_mixture_em():
    # 2 clusters in 2D
    np.random.seed(42)
    c1 = np.random.normal(loc=[2.0, 2.0], scale=0.5, size=(50, 2))
    c2 = np.random.normal(loc=[-2.0, -2.0], scale=0.5, size=(50, 2))
    X = np.vstack([c1, c2])

    gmm = GaussianMixtureEM(n_components=2, max_iter=20, seed=42)
    gmm.fit(X)

    assert len(gmm.weights) == 2
    assert np.isclose(np.sum(gmm.weights), 1.0)
    assert gmm.means.shape == (2, 2)

    labels = gmm.predict(X)
    assert len(labels) == 100
    # Cluster separation check: first 50 and last 50 should have different labels
    label_c1 = np.bincount(labels[:50]).argmax()
    label_c2 = np.bincount(labels[50:]).argmax()
    assert label_c1 != label_c2
