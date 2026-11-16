"""
Bayesian Network Implementation
-------------------------------
Implements:
1. DiscreteFactor:
   - High-performance discrete factor representation for joint and conditional distributions.
   - Core operations: Factor Product, Marginalization (Summing Out), Reduction (Evidence Conditioning), Normalization.
2. BayesianNetwork:
   - Directed Acyclic Graph (DAG) with cycle detection and topological sorting.
   - Conditional Probability Tables (CPTs) specification.
   - Exact Probabilistic Inference using the Variable Elimination Algorithm.
   - Prior / Forward sampling for stochastic validation.
"""

from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np


class DiscreteFactor:
    """
    Represents a factor phi(X_1, ..., X_k) over discrete random variables.
    """

    def __init__(
        self,
        variables: List[str],
        cardinalities: Dict[str, int],
        values: np.ndarray,
        state_names: Optional[Dict[str, List[str]]] = None,
    ):
        self.variables = list(variables)
        self.cardinalities = {v: cardinalities[v] for v in self.variables}
        self.shape = tuple(self.cardinalities[v] for v in self.variables)
        val_arr = np.array(values, dtype=np.float64)
        if len(self.shape) > 0:
            self.values = val_arr.reshape(self.shape)
        else:
            self.values = val_arr.reshape(())
        self.state_names = state_names or {}

    def copy(self) -> "DiscreteFactor":
        return DiscreteFactor(
            variables=list(self.variables),
            cardinalities=dict(self.cardinalities),
            values=self.values.copy(),
            state_names={k: list(v) for k, v in self.state_names.items()},
        )

    def reduce(self, var: str, value_index: int) -> "DiscreteFactor":
        """
        Condition on evidence: Instantiate variable `var` to `value_index`.
        Reduces dimensionality by 1.
        """
        if var not in self.variables:
            return self.copy()

        axis = self.variables.index(var)
        sliced = np.take(self.values, indices=value_index, axis=axis)

        new_vars = [v for v in self.variables if v != var]
        new_cards = {v: self.cardinalities[v] for v in new_vars}
        new_state_names = {k: v for k, v in self.state_names.items() if k != var}

        return DiscreteFactor(new_vars, new_cards, sliced, new_state_names)

    def product(self, other: "DiscreteFactor") -> "DiscreteFactor":
        """
        Factor product phi_1 * phi_2 with rigorous multidimensional axis alignment.
        """
        # Handle scalar factors
        if len(self.variables) == 0:
            scale = float(self.values.item() if self.values.size == 1 else np.prod(self.values))
            res = other.copy()
            res.values = res.values * scale
            return res
        if len(other.variables) == 0:
            scale = float(other.values.item() if other.values.size == 1 else np.prod(other.values))
            res = self.copy()
            res.values = res.values * scale
            return res

        combined_vars = list(self.variables)
        for v in other.variables:
            if v not in combined_vars:
                combined_vars.append(v)

        combined_cards = {
            v: self.cardinalities.get(v, other.cardinalities.get(v, 1))
            for v in combined_vars
        }
        combined_state_names = {**self.state_names, **other.state_names}

        # Align self.values to combined_vars
        val1 = self.values
        for v in combined_vars:
            if v not in self.variables:
                val1 = np.expand_dims(val1, axis=-1)
        current_vars1 = list(self.variables) + [v for v in combined_vars if v not in self.variables]
        perm1 = [current_vars1.index(v) for v in combined_vars]
        if perm1:
            val1 = np.transpose(val1, perm1)

        # Align other.values to combined_vars
        val2 = other.values
        for v in combined_vars:
            if v not in other.variables:
                val2 = np.expand_dims(val2, axis=-1)
        current_vars2 = list(other.variables) + [v for v in combined_vars if v not in other.variables]
        perm2 = [current_vars2.index(v) for v in combined_vars]
        if perm2:
            val2 = np.transpose(val2, perm2)

        # Element-wise product across aligned dimensions
        new_values = val1 * val2
        return DiscreteFactor(combined_vars, combined_cards, new_values, combined_state_names)

    def marginalize(self, var: str) -> "DiscreteFactor":
        """
        Sum out variable `var` from this factor: sum_{var} phi.
        """
        if var not in self.variables:
            return self.copy()

        axis = self.variables.index(var)
        new_values = np.sum(self.values, axis=axis)

        new_vars = [v for v in self.variables if v != var]
        new_cards = {v: self.cardinalities[v] for v in new_vars}
        new_state_names = {k: v for k, v in self.state_names.items() if k != var}

        return DiscreteFactor(new_vars, new_cards, new_values, new_state_names)

    def normalize(self) -> "DiscreteFactor":
        """Normalize factor so its values sum to 1.0."""
        total = np.sum(self.values)
        if total > 0:
            self.values = self.values / total
        return self


class BayesianNetwork:
    """
    Bayesian Network with Directed Acyclic Graph topology and
    Exact Inference via Variable Elimination.
    """

    def __init__(self):
        self.nodes: List[str] = []
        self.node_states: Dict[str, List[str]] = {}
        self.parents: Dict[str, List[str]] = {}
        self.children: Dict[str, List[str]] = {}
        self.cpts: Dict[str, DiscreteFactor] = {}

    def add_node(self, name: str, states: List[str]) -> None:
        """Add a random variable node and its possible categorical states."""
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists.")
        self.nodes.append(name)
        self.node_states[name] = list(states)
        self.parents[name] = []
        self.children[name] = []

    def add_edge(self, parent: str, child: str) -> None:
        """Add a directed edge parent -> child."""
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("Both parent and child must be existing nodes.")
        if child not in self.children[parent]:
            self.children[parent].append(child)
            self.parents[child].append(parent)

        if not self._is_acyclic():
            # Undo edge addition
            self.children[parent].remove(child)
            self.parents[child].remove(parent)
            raise ValueError(f"Adding edge {parent} -> {child} introduces a cycle!")

    def _is_acyclic(self) -> bool:
        """Check if graph is DAG using Kahn's algorithm."""
        in_degree = {n: len(self.parents[n]) for n in self.nodes}
        queue = [n for n in self.nodes if in_degree[n] == 0]
        visited = 0

        while queue:
            curr = queue.pop(0)
            visited += 1
            for child in self.children[curr]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return visited == len(self.nodes)

    def set_cpt(self, node: str, cpt_array: np.ndarray) -> None:
        """
        Define the Conditional Probability Table for `node`.
        The factor dimensions order is [*parents, node].
        """
        if node not in self.nodes:
            raise ValueError(f"Unknown node {node}")

        vars_order = list(self.parents[node]) + [node]
        cardinalities = {v: len(self.node_states[v]) for v in vars_order}
        state_names = {v: self.node_states[v] for v in vars_order}

        factor = DiscreteFactor(vars_order, cardinalities, cpt_array, state_names)
        self.cpts[node] = factor

    def variable_elimination(
        self,
        query_var: str,
        evidence: Optional[Dict[str, str]] = None,
        elimination_order: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Exact Inference using the Variable Elimination Algorithm.

        Computes:
            P(query_var | evidence)

        Args:
            query_var: Target random variable to compute marginal posterior.
            evidence: Dictionary of observed variables and their state names {var: state}.
            elimination_order: Optional variable elimination ordering.

        Returns:
            Dictionary mapping each state of query_var to its posterior probability.
        """
        if query_var not in self.nodes:
            raise ValueError(f"Query variable '{query_var}' not in network.")

        evidence = evidence or {}

        # 1. Initialize factor pool from all node CPTs
        active_factors = [factor.copy() for factor in self.cpts.values()]

        # 2. Instantiate evidence by reducing factors
        for ev_var, ev_val in evidence.items():
            if ev_var not in self.nodes:
                continue
            if ev_var == query_var:
                raise ValueError("Query variable cannot be part of observed evidence.")
            val_idx = self.node_states[ev_var].index(ev_val)
            active_factors = [f.reduce(ev_var, val_idx) for f in active_factors]

        # 3. Determine elimination order for unobserved, non-query hidden variables
        hidden_vars = [
            n for n in self.nodes if n != query_var and n not in evidence
        ]

        if elimination_order is None:
            # Min-neighbors heuristic for elimination order
            elimination_order = sorted(
                hidden_vars,
                key=lambda v: sum(1 for f in active_factors if v in f.variables),
            )

        # 4. Eliminate hidden variables sequentially
        for var in elimination_order:
            factors_with_var = [f for f in active_factors if var in f.variables]
            factors_without_var = [f for f in active_factors if var not in f.variables]

            if not factors_with_var:
                continue

            # Multiply all factors containing var
            combined = factors_with_var[0]
            for f in factors_with_var[1:]:
                combined = combined.product(f)

            # Sum out var
            summed_out = combined.marginalize(var)

            # Update active factors pool
            active_factors = factors_without_var
            if len(summed_out.variables) > 0 or summed_out.values.size > 0:
                active_factors.append(summed_out)

        # 5. Multiply remaining factors containing query_var
        relevant_factors = [f for f in active_factors if query_var in f.variables]
        if not relevant_factors:
            states = self.node_states[query_var]
            return {s: 1.0 / len(states) for s in states}

        final_factor = relevant_factors[0]
        for f in relevant_factors[1:]:
            final_factor = final_factor.product(f)

        final_factor.normalize()

        # Extract output mapping
        states = self.node_states[query_var]
        flat_values = final_factor.values.flatten()
        return {states[i]: float(flat_values[i]) for i in range(len(states))}

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order."""
        in_degree = {n: len(self.parents[n]) for n in self.nodes}
        queue = [n for n in self.nodes if in_degree[n] == 0]
        order = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for child in self.children[curr]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return order

    def sample_forward(self, num_samples: int = 1000, seed: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Generate forward samples from the network joint distribution.
        """
        rng = np.random.default_rng(seed)
        order = self.topological_sort()
        samples = []

        for _ in range(num_samples):
            sample: Dict[str, str] = {}
            for node in order:
                factor = self.cpts[node]
                cond_factor = factor.copy()
                for parent in self.parents[node]:
                    val = sample[parent]
                    val_idx = self.node_states[parent].index(val)
                    cond_factor = cond_factor.reduce(parent, val_idx)

                cond_factor.normalize()
                probs = cond_factor.values.flatten()
                chosen_idx = rng.choice(len(self.node_states[node]), p=probs)
                sample[node] = self.node_states[node][chosen_idx]
            samples.append(sample)

        return samples

    def to_dict(self) -> Dict[str, Any]:
        """Serialize network metadata and CPTs to dictionary."""
        return {
            "nodes": self.nodes,
            "node_states": self.node_states,
            "parents": self.parents,
            "children": self.children,
            "cpts": {
                node: {
                    "variables": self.cpts[node].variables,
                    "values": self.cpts[node].values.tolist(),
                }
                for node in self.nodes
                if node in self.cpts
            },
        }
