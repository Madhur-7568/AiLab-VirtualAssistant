"""
Hidden Markov Model (HMM) Implementation
----------------------------------------
Implements a first-order discrete Hidden Markov Model lambda = (S, V, A, B, pi):
- S: Finite set of hidden states {s_1, ..., s_N}
- V: Finite set of discrete observations {v_1, ..., v_M}
- A: Transition probability matrix, A[i, j] = P(q_{t+1} = s_j | q_t = s_i)
- B: Emission probability matrix, B[j, k] = P(o_t = v_k | q_t = s_j)
- pi: Initial state distribution, pi[i] = P(q_1 = s_i)

Includes:
- Forward Algorithm with scaling for numerical stability
- Backward Algorithm with scaling
- Viterbi Algorithm in log-space for optimal sequence decoding
- Posterior state decoding gamma_t(i)
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class HiddenMarkovModel:
    def __init__(
        self,
        states: List[str],
        observations: List[str],
        start_prob: Optional[np.ndarray] = None,
        trans_prob: Optional[np.ndarray] = None,
        emission_prob: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Hidden Markov Model.

        Args:
            states: List of hidden state names (size N).
            observations: List of observation token vocabulary (size M).
            start_prob: Prior probability vector pi (shape: (N,)).
            trans_prob: State transition matrix A (shape: (N, N)).
            emission_prob: Observation emission matrix B (shape: (N, M)).
            seed: Optional random seed for reproducible initialization.
        """
        self.states = list(states)
        self.observations = list(observations)
        self.num_states = len(self.states)
        self.num_obs = len(self.observations)

        self.state_to_idx: Dict[str, int] = {s: i for i, s in enumerate(self.states)}
        self.obs_to_idx: Dict[str, int] = {o: i for i, o in enumerate(self.observations)}

        rng = np.random.default_rng(seed)

        # 1. Initial State Probabilities pi
        if start_prob is not None:
            self.pi = np.array(start_prob, dtype=np.float64)
            self._validate_stochastic_vector(self.pi, "start_prob")
        else:
            uniform = rng.uniform(0.5, 1.5, size=self.num_states)
            self.pi = uniform / np.sum(uniform)

        # 2. Transition Probability Matrix A
        if trans_prob is not None:
            self.A = np.array(trans_prob, dtype=np.float64)
            self._validate_stochastic_matrix(self.A, "trans_prob")
        else:
            uniform_a = rng.uniform(0.5, 1.5, size=(self.num_states, self.num_states))
            self.A = uniform_a / np.sum(uniform_a, axis=1, keepdims=True)

        # 3. Emission Probability Matrix B
        if emission_prob is not None:
            self.B = np.array(emission_prob, dtype=np.float64)
            self._validate_stochastic_matrix(self.B, "emission_prob")
        else:
            uniform_b = rng.uniform(0.5, 1.5, size=(self.num_states, self.num_obs))
            self.B = uniform_b / np.sum(uniform_b, axis=1, keepdims=True)

    @staticmethod
    def _validate_stochastic_vector(vec: np.ndarray, name: str) -> None:
        if not np.all(vec >= 0):
            raise ValueError(f"{name} must contain non-negative probabilities.")
        if not np.isclose(np.sum(vec), 1.0, atol=1e-4):
            raise ValueError(f"{name} must sum to 1.0 (got {np.sum(vec):.4f}).")

    @staticmethod
    def _validate_stochastic_matrix(mat: np.ndarray, name: str) -> None:
        if not np.all(mat >= 0):
            raise ValueError(f"{name} must contain non-negative probabilities.")
        row_sums = np.sum(mat, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            raise ValueError(f"Every row of {name} must sum to 1.0.")

    def encode_observations(self, obs_seq: List[str], unknown_strategy: str = "uniform") -> List[int]:
        """Convert string observation sequence to vocabulary indices."""
        indices = []
        for o in obs_seq:
            if o in self.obs_to_idx:
                indices.append(self.obs_to_idx[o])
            else:
                # Handle unknown tokens: map to special token or fallback
                if "<UNK>" in self.obs_to_idx:
                    indices.append(self.obs_to_idx["<UNK>"])
                else:
                    # Fallback to closest match or first index
                    indices.append(0)
        return indices

    def forward(self, obs_seq: List[int]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward Algorithm with scaling to prevent numerical underflow.

        Computes:
            alpha_hat[t, i] = P(q_t = s_i | o_1, ..., o_t, lambda)
            scale[t] = 1 / sum_i(alpha[t, i])
            log_likelihood = -sum(log(scale[t]))

        Returns:
            alpha_hat: Scaled forward variables of shape (T, N).
            scale: Scaling factors vector of length T.
            log_likelihood: log P(O | lambda).
        """
        T = len(obs_seq)
        if T == 0:
            return np.empty((0, self.num_states)), np.empty(0), 0.0

        alpha = np.zeros((T, self.num_states), dtype=np.float64)
        scale = np.zeros(T, dtype=np.float64)

        # Base case t = 0
        o_0 = obs_seq[0]
        alpha[0, :] = self.pi * self.B[:, o_0]
        sum_0 = np.sum(alpha[0, :])
        if sum_0 == 0:
            # Add numerical stabilizer
            alpha[0, :] += 1e-12
            sum_0 = np.sum(alpha[0, :])
        scale[0] = 1.0 / sum_0
        alpha[0, :] *= scale[0]

        # Induction t = 1 ... T-1
        for t in range(1, T):
            o_t = obs_seq[t]
            # alpha[t, j] = [sum_i alpha[t-1, i] * A[i, j]] * B[j, o_t]
            alpha[t, :] = np.dot(alpha[t - 1, :], self.A) * self.B[:, o_t]
            sum_t = np.sum(alpha[t, :])
            if sum_t == 0:
                alpha[t, :] += 1e-12
                sum_t = np.sum(alpha[t, :])
            scale[t] = 1.0 / sum_t
            alpha[t, :] *= scale[t]

        log_likelihood = -np.sum(np.log(scale + 1e-300))
        return alpha, scale, float(log_likelihood)

    def backward(self, obs_seq: List[int], scale: np.ndarray) -> np.ndarray:
        """
        Backward Algorithm using the scaling factors from the Forward pass.

        Computes:
            beta_hat[t, i] for t = T-1 down to 0.

        Returns:
            beta_hat: Scaled backward variables of shape (T, N).
        """
        T = len(obs_seq)
        if T == 0:
            return np.empty((0, self.num_states))

        beta = np.zeros((T, self.num_states), dtype=np.float64)

        # Base case t = T-1
        beta[T - 1, :] = scale[T - 1]

        # Induction t = T-2 down to 0
        for t in range(T - 2, -1, -1):
            o_next = obs_seq[t + 1]
            # beta[t, i] = sum_j A[i, j] * B[j, o_{t+1}] * beta[t+1, j]
            beta[t, :] = np.dot(self.A, self.B[:, o_next] * beta[t + 1, :])
            beta[t, :] *= scale[t]

        return beta

    def viterbi(self, obs_seq: List[int]) -> Tuple[List[str], float, np.ndarray]:
        """
        Viterbi Algorithm in log-space.
        Finds the globally optimal hidden state sequence Q* that maximizes P(Q, O | lambda).

        Returns:
            best_states: List of decoded state names [q_1*, ..., q_T*].
            best_log_prob: Log probability of the best path.
            trellis: Log-probability Viterbi trellis matrix (T, N).
        """
        T = len(obs_seq)
        if T == 0:
            return [], -float("inf"), np.empty((0, self.num_states))

        log_pi = np.log(np.maximum(self.pi, 1e-300))
        log_A = np.log(np.maximum(self.A, 1e-300))
        log_B = np.log(np.maximum(self.B, 1e-300))

        viterbi_mat = np.zeros((T, self.num_states), dtype=np.float64)
        backpointer = np.zeros((T, self.num_states), dtype=np.int32)

        # Initialization (t = 0)
        o_0 = obs_seq[0]
        viterbi_mat[0, :] = log_pi + log_B[:, o_0]

        # Recursion (t = 1 to T-1)
        for t in range(1, T):
            o_t = obs_seq[t]
            # For each state j: max_i (viterbi_mat[t-1, i] + log_A[i, j]) + log_B[j, o_t]
            prev_paths = viterbi_mat[t - 1, :, None] + log_A  # shape (N, N)
            backpointer[t, :] = np.argmax(prev_paths, axis=0)
            viterbi_mat[t, :] = np.max(prev_paths, axis=0) + log_B[:, o_t]

        # Termination
        best_last_state = int(np.argmax(viterbi_mat[T - 1, :]))
        best_log_prob = float(viterbi_mat[T - 1, best_last_state])

        # Backtracking path
        best_path_indices = [best_last_state]
        for t in range(T - 1, 0, -1):
            best_last_state = int(backpointer[t, best_last_state])
            best_path_indices.append(best_last_state)

        best_path_indices.reverse()
        best_states = [self.states[idx] for idx in best_path_indices]

        return best_states, best_log_prob, viterbi_mat

    def posterior_decoding(self, obs_seq: List[int]) -> np.ndarray:
        """
        Compute posterior state probabilities gamma_t(i) = P(q_t = s_i | O, lambda).

        Returns:
            gamma: Matrix of shape (T, N) where each row sums to 1.0.
        """
        alpha, scale, _ = self.forward(obs_seq)
        beta = self.backward(obs_seq, scale)

        # gamma[t, i] = alpha_hat[t, i] * beta_hat[t, i] / scale[t]
        gamma = alpha * beta / scale[:, None]
        # Normalize rows to protect from minor floating point inaccuracies
        row_sums = np.sum(gamma, axis=1, keepdims=True)
        gamma = np.divide(gamma, row_sums, out=gamma, where=row_sums > 0)
        return gamma

    def sample(self, length: int, seed: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        Generate a synthetic sequence of hidden states and observations.
        """
        rng = np.random.default_rng(seed)
        states_seq = []
        obs_seq = []

        # Sample initial state
        curr_state = rng.choice(self.num_states, p=self.pi)
        for _ in range(length):
            states_seq.append(self.states[curr_state])
            # Sample observation from state
            o = rng.choice(self.num_obs, p=self.B[curr_state, :])
            obs_seq.append(self.observations[o])
            # Transition to next state
            curr_state = rng.choice(self.num_states, p=self.A[curr_state, :])

        return states_seq, obs_seq

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model parameters to JSON-serializable dict."""
        return {
            "states": self.states,
            "observations": self.observations,
            "start_prob": self.pi.tolist(),
            "trans_prob": self.A.tolist(),
            "emission_prob": self.B.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HiddenMarkovModel":
        """Reconstruct model from dictionary."""
        return cls(
            states=data["states"],
            observations=data["observations"],
            start_prob=np.array(data["start_prob"]),
            trans_prob=np.array(data["trans_prob"]),
            emission_prob=np.array(data["emission_prob"]),
        )
