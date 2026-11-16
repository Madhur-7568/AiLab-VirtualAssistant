"""
Unit tests for Hidden Markov Model (HMM)
Tests Forward algorithm, Backward algorithm, Viterbi decoding, and posterior probabilities.
"""

import pytest
import numpy as np
from src.models.hidden_markov_model import HiddenMarkovModel


@pytest.fixture
def simple_hmm():
    """
    Classic 2-state Rainy/Sunny weather model.
    States: Rainy (0), Sunny (1)
    Observations: Walk (0), Shop (1), Clean (2)
    """
    states = ["Rainy", "Sunny"]
    observations = ["walk", "shop", "clean"]
    pi = np.array([0.6, 0.4])
    A = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    B = np.array([
        [0.1, 0.4, 0.5],
        [0.6, 0.3, 0.1]
    ])
    return HiddenMarkovModel(states, observations, pi, A, B)


def test_hmm_initialization(simple_hmm):
    assert simple_hmm.num_states == 2
    assert simple_hmm.num_obs == 3
    assert np.isclose(np.sum(simple_hmm.pi), 1.0)
    assert np.allclose(np.sum(simple_hmm.A, axis=1), 1.0)
    assert np.allclose(np.sum(simple_hmm.B, axis=1), 1.0)


def test_forward_algorithm(simple_hmm):
    # Observation: walk (0), clean (2), shop (1)
    obs = [0, 2, 1]
    alpha, scale, log_lik = simple_hmm.forward(obs)

    assert alpha.shape == (3, 2)
    assert len(scale) == 3
    # Log likelihood should be a valid negative finite number
    assert np.isfinite(log_lik)
    assert log_lik < 0.0


def test_backward_algorithm(simple_hmm):
    obs = [0, 2, 1]
    alpha, scale, log_lik = simple_hmm.forward(obs)
    beta = simple_hmm.backward(obs, scale)

    assert beta.shape == (3, 2)
    assert np.all(np.isfinite(beta))


def test_viterbi_decoding(simple_hmm):
    # Walk (0), shop (1), clean (2)
    obs = [0, 1, 2]
    best_states, log_prob, trellis = simple_hmm.viterbi(obs)

    assert len(best_states) == 3
    assert all(s in ["Rainy", "Sunny"] for s in best_states)
    assert np.isfinite(log_prob)
    # Walk most likely emitted by Sunny, Clean most likely emitted by Rainy
    assert best_states[0] == "Sunny"
    assert best_states[2] == "Rainy"


def test_posterior_decoding(simple_hmm):
    obs = [0, 2]
    gamma = simple_hmm.posterior_decoding(obs)

    assert gamma.shape == (2, 2)
    # Each time step posterior over states must sum to 1.0
    assert np.allclose(np.sum(gamma, axis=1), 1.0)
