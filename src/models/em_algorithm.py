"""
Expectation-Maximization (EM) Algorithm Implementation
-----------------------------------------------------
Includes:
1. BaumWelchLearner:
   - Unsupervised training of Hidden Markov Model parameters (A, B, pi)
     from unannotated observation sequences.
   - E-step: Computes state occupancies gamma_t(i) and transition occupancies xi_t(i, j).
   - M-step: Re-estimates pi, A, B with Laplace smoothing.
   - Convergence monitoring via total log-likelihood.
2. GaussianMixtureEM:
   - EM algorithm for continuous Gaussian Mixture Models (clustering user intents/embeddings).
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from src.models.hidden_markov_model import HiddenMarkovModel


class BaumWelchLearner:
    """
    Implements the Baum-Welch (EM) algorithm for Hidden Markov Models.
    Learns lambda* = argmax_lambda P(O_1, ..., O_K | lambda).
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-4,
        pseudocount: float = 1e-3,
        verbose: bool = False,
    ):
        """
        Args:
            max_iter: Maximum EM iterations.
            tol: Convergence tolerance for change in log-likelihood.
            pseudocount: Smoothing parameter (Laplace/Dirichlet) for M-step.
            verbose: Whether to print iteration progress.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.pseudocount = pseudocount
        self.verbose = verbose
        self.log_likelihood_history: List[float] = []

    def fit(
        self,
        hmm: HiddenMarkovModel,
        sequences: List[List[int]],
    ) -> HiddenMarkovModel:
        """
        Fit HMM parameters on a collection of observation sequences using EM.

        Args:
            hmm: Initialized HiddenMarkovModel instance to train.
            sequences: List of observation sequences (encoded as vocabulary indices).

        Returns:
            Trained HiddenMarkovModel with updated pi, A, and B matrices.
        """
        N = hmm.num_states
        M = hmm.num_obs
        self.log_likelihood_history = []

        for iteration in range(self.max_iter):
            # Accumulators for the M-step
            pi_num = np.zeros(N, dtype=np.float64)
            A_num = np.zeros((N, N), dtype=np.float64)
            A_den = np.zeros(N, dtype=np.float64)
            B_num = np.zeros((N, M), dtype=np.float64)
            B_den = np.zeros(N, dtype=np.float64)

            total_log_likelihood = 0.0

            # ---------------------------------------------
            # E-STEP: Compute expectations across all sequences
            # ---------------------------------------------
            for seq in sequences:
                T = len(seq)
                if T < 2:
                    continue

                alpha, scale, log_lik = hmm.forward(seq)
                beta = hmm.backward(seq, scale)
                total_log_likelihood += log_lik

                # gamma[t, i] = P(q_t = s_i | O, lambda)
                # alpha_hat[t, i] * beta_hat[t, i] / scale[t]
                gamma = (alpha * beta) / scale[:, None]
                # Normalize row-wise
                gamma = np.divide(
                    gamma,
                    np.sum(gamma, axis=1, keepdims=True) + 1e-300,
                )

                # xi[t, i, j] = P(q_t = s_i, q_{t+1} = s_j | O, lambda)
                xi = np.zeros((T - 1, N, N), dtype=np.float64)
                for t in range(T - 1):
                    o_next = seq[t + 1]
                    # alpha[t, i] * A[i, j] * B[j, o_{t+1}] * beta[t+1, j]
                    trans_mat = hmm.A * (hmm.B[:, o_next] * beta[t + 1, :])[None, :]
                    xi_t = alpha[t, :, None] * trans_mat
                    denom = np.sum(xi_t)
                    if denom > 0:
                        xi[t, :, :] = xi_t / denom
                    else:
                        xi[t, :, :] = 1.0 / (N * N)

                # Accumulate expectations
                pi_num += gamma[0, :]
                A_num += np.sum(xi, axis=0)
                A_den += np.sum(gamma[:-1, :], axis=0)

                for k in range(M):
                    mask = (np.array(seq) == k)
                    if np.any(mask):
                        B_num[:, k] += np.sum(gamma[mask, :], axis=0)
                B_den += np.sum(gamma, axis=0)

            self.log_likelihood_history.append(total_log_likelihood)

            # ---------------------------------------------
            # M-STEP: Maximize parameters with Laplace smoothing
            # ---------------------------------------------
            num_seqs = max(len(sequences), 1)
            # Re-estimate initial distribution pi
            new_pi = (pi_num + self.pseudocount) / (num_seqs + N * self.pseudocount)
            new_pi = new_pi / np.sum(new_pi)

            # Re-estimate transition matrix A
            new_A = (A_num + self.pseudocount) / (A_den[:, None] + N * self.pseudocount)
            new_A = new_A / np.sum(new_A, axis=1, keepdims=True)

            # Re-estimate emission matrix B
            new_B = (B_num + self.pseudocount) / (B_den[:, None] + M * self.pseudocount)
            new_B = new_B / np.sum(new_B, axis=1, keepdims=True)

            hmm.pi = new_pi
            hmm.A = new_A
            hmm.B = new_B

            if self.verbose:
                print(f"[Baum-Welch] Iter {iteration + 1}/{self.max_iter} - Log-Likelihood: {total_log_likelihood:.4f}")

            # Check convergence
            if iteration > 0:
                delta = abs(total_log_likelihood - self.log_likelihood_history[-2])
                if delta < self.tol:
                    if self.verbose:
                        print(f"[Baum-Welch] Converged at iteration {iteration + 1} with delta {delta:.6f}")
                    break

        return hmm


class GaussianMixtureEM:
    """
    Gaussian Mixture Model trained via the Expectation-Maximization algorithm.
    Used to cluster conversational query feature vectors into latent intent clusters.
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 50,
        tol: float = 1e-4,
        reg_covar: float = 1e-5,
        seed: Optional[int] = None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.seed = seed

        self.weights: np.ndarray = np.empty(0)
        self.means: np.ndarray = np.empty((0, 0))
        self.covariances: np.ndarray = np.empty((0, 0, 0))
        self.log_likelihood_history: List[float] = []

    def fit(self, X: np.ndarray) -> "GaussianMixtureEM":
        """
        Fit GMM on dataset X (shape: N samples x D dimensions) using EM.
        """
        rng = np.random.default_rng(self.seed)
        N, D = X.shape
        K = self.n_components

        # Initialize weights uniformly
        self.weights = np.ones(K) / K

        # Initialize means randomly from data points
        indices = rng.choice(N, size=K, replace=False if N >= K else True)
        self.means = X[indices].copy()

        # Initialize covariances as sample covariance
        sample_cov = np.cov(X, rowvar=False) if N > 1 else np.eye(D)
        if sample_cov.ndim == 0:
            sample_cov = np.array([[sample_cov]])
        self.covariances = np.array([sample_cov + np.eye(D) * self.reg_covar for _ in range(K)])

        self.log_likelihood_history = []

        for it in range(self.max_iter):
            # ---------------------------------------------
            # E-step: Compute responsibilities r[i, k] = P(cluster k | x_i)
            # ---------------------------------------------
            likelihoods = np.zeros((N, K))
            for k in range(K):
                diff = X - self.means[k]
                cov = self.covariances[k] + np.eye(D) * self.reg_covar
                try:
                    inv_cov = np.linalg.pinv(cov)
                    det_cov = np.linalg.det(cov)
                    det_cov = max(det_cov, 1e-300)
                except np.linalg.LinAlgError:
                    inv_cov = np.eye(D)
                    det_cov = 1.0

                norm_const = 1.0 / np.sqrt(((2 * np.pi) ** D) * det_cov)
                # Mahalanobis distance
                dist = np.sum(diff @ inv_cov * diff, axis=1)
                likelihoods[:, k] = self.weights[k] * norm_const * np.exp(-0.5 * np.clip(dist, 0, 700))

            sum_likelihood = np.sum(likelihoods, axis=1, keepdims=True)
            log_lik = float(np.sum(np.log(np.maximum(sum_likelihood, 1e-300))))
            self.log_likelihood_history.append(log_lik)

            responsibilities = likelihoods / np.maximum(sum_likelihood, 1e-300)

            # ---------------------------------------------
            # M-step: Update weights, means, and covariances
            # ---------------------------------------------
            Nk = np.sum(responsibilities, axis=0) + 1e-12
            self.weights = Nk / N

            for k in range(K):
                # Update mean
                self.means[k] = np.sum(responsibilities[:, k, None] * X, axis=0) / Nk[k]
                # Update covariance
                diff = X - self.means[k]
                cov_k = (responsibilities[:, k, None, None] * (diff[:, :, None] @ diff[:, None, :]))
                self.covariances[k] = np.sum(cov_k, axis=0) / Nk[k] + np.eye(D) * self.reg_covar

            if it > 0 and abs(log_lik - self.log_likelihood_history[-2]) < self.tol:
                break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior probabilities of cluster assignment."""
        N, D = X.shape
        K = self.n_components
        likelihoods = np.zeros((N, K))
        for k in range(K):
            diff = X - self.means[k]
            cov = self.covariances[k] + np.eye(D) * self.reg_covar
            inv_cov = np.linalg.pinv(cov)
            det_cov = max(np.linalg.det(cov), 1e-300)
            norm_const = 1.0 / np.sqrt(((2 * np.pi) ** D) * det_cov)
            dist = np.sum(diff @ inv_cov * diff, axis=1)
            likelihoods[:, k] = self.weights[k] * norm_const * np.exp(-0.5 * np.clip(dist, 0, 700))

        sum_lik = np.sum(likelihoods, axis=1, keepdims=True)
        return likelihoods / np.maximum(sum_lik, 1e-300)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign data points to highest-responsibility cluster."""
        return np.argmax(self.predict_proba(X), axis=1)
