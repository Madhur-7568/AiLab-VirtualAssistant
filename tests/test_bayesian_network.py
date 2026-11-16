"""
Unit tests for Bayesian Network and Variable Elimination
Tests DAG validation, factor operations (product, marginalization, reduction),
and exact posterior inference via Variable Elimination.
"""

import pytest
import numpy as np
from src.models.bayesian_network import BayesianNetwork, DiscreteFactor


def test_factor_operations():
    # Variable A with 2 states, B with 2 states
    f1 = DiscreteFactor(["A"], {"A": 2}, np.array([0.6, 0.4]))
    f2 = DiscreteFactor(["A", "B"], {"A": 2, "B": 2}, np.array([[0.7, 0.3], [0.2, 0.8]]))

    # Factor Product
    f_prod = f1.product(f2)
    assert f_prod.variables == ["A", "B"]
    assert np.isclose(f_prod.values[0, 0], 0.6 * 0.7)
    assert np.isclose(f_prod.values[1, 1], 0.4 * 0.8)

    # Marginalization (sum out A)
    f_marg = f_prod.marginalize("A")
    assert f_marg.variables == ["B"]
    # P(B=0) = 0.6*0.7 + 0.4*0.2 = 0.42 + 0.08 = 0.50
    assert np.isclose(f_marg.values[0], 0.50)
    # P(B=1) = 0.6*0.3 + 0.4*0.8 = 0.18 + 0.32 = 0.50
    assert np.isclose(f_marg.values[1], 0.50)

    # Reduction (conditioning B=0)
    f_red = f_prod.reduce("B", 0)
    assert f_red.variables == ["A"]
    assert np.isclose(f_red.values[0], 0.42)
    assert np.isclose(f_red.values[1], 0.08)


def test_bayesian_network_cycle_rejection():
    bn = BayesianNetwork()
    bn.add_node("A", ["T", "F"])
    bn.add_node("B", ["T", "F"])
    bn.add_node("C", ["T", "F"])

    bn.add_edge("A", "B")
    bn.add_edge("B", "C")
    # Adding C -> A should introduce a cycle and raise ValueError
    with pytest.raises(ValueError):
        bn.add_edge("C", "A")


def test_alarm_network_exact_inference():
    """
    Classic Judea Pearl Alarm Network:
    Burglary (B), Earthquake (E) -> Alarm (A) -> JohnCalls (J), MaryCalls (M)
    """
    bn = BayesianNetwork()
    for node in ["B", "E", "A", "J", "M"]:
        bn.add_node(node, ["True", "False"])

    bn.add_edge("B", "A")
    bn.add_edge("E", "A")
    bn.add_edge("A", "J")
    bn.add_edge("A", "M")

    # P(B)
    bn.set_cpt("B", np.array([0.001, 0.999]))
    # P(E)
    bn.set_cpt("E", np.array([0.002, 0.998]))
    # P(A | B, E) -> shape (2, 2, 2)
    # B=T, E=T -> [0.95, 0.05]
    # B=T, E=F -> [0.94, 0.06]
    # B=F, E=T -> [0.29, 0.71]
    # B=F, E=F -> [0.001, 0.999]
    cpt_a = np.array([
        [[0.95, 0.05], [0.94, 0.06]],
        [[0.29, 0.71], [0.001, 0.999]],
    ])
    bn.set_cpt("A", cpt_a)

    # P(J | A)
    # A=T -> [0.90, 0.10], A=F -> [0.05, 0.95]
    cpt_j = np.array([[0.90, 0.10], [0.05, 0.95]])
    bn.set_cpt("J", cpt_j)

    # P(M | A)
    # A=T -> [0.70, 0.30], A=F -> [0.01, 0.99]
    cpt_m = np.array([[0.70, 0.30], [0.01, 0.99]])
    bn.set_cpt("M", cpt_m)

    # Query P(B | J=True, M=True)
    posterior = bn.variable_elimination("B", evidence={"J": "True", "M": "True"})

    assert "True" in posterior and "False" in posterior
    assert np.isclose(posterior["True"] + posterior["False"], 1.0)
    # Known benchmark: P(B=True | J=True, M=True) is approx 0.284
    assert 0.25 <= posterior["True"] <= 0.35
