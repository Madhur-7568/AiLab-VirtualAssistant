"""
Bayesian Decision Engine for Virtual Assistant
----------------------------------------------
Implements a Bayesian Network to reason under uncertainty about user intent,
context, and the optimal assistant action.
"""

from typing import Dict, Any, Optional
import numpy as np
from src.models.bayesian_network import BayesianNetwork


class BayesianDecisionEngine:
    """
    Constructs and queries the Virtual Assistant's Bayesian Network.
    Topology:
        TimeOfDay ---------------------> UserFatigue
        UserSentiment -----------------> UserFatigue
        UserFatigue -------------------> RecommendedAction
        UserUrgency -------------------> TaskImportance
        DetectedIntent ----------------> TaskImportance
        DetectedIntent ----------------> RecommendedAction
        TaskImportance ----------------> RecommendedAction
        RecommendedAction -------------> RequireConfirmation
    """

    def __init__(self):
        self.bn = BayesianNetwork()
        self._build_network()

    def _build_network(self) -> None:
        bn = self.bn

        # 1. Add nodes and state domains
        bn.add_node("TimeOfDay", ["Morning", "Afternoon", "Evening", "Night"])
        bn.add_node("UserUrgency", ["Normal", "High", "Critical"])
        bn.add_node("UserSentiment", ["Calm", "Frustrated", "Busy"])
        bn.add_node("DetectedIntent", ["Schedule", "Reminder", "Query", "HealthCheck", "Greeting"])
        bn.add_node("TaskImportance", ["Normal", "High", "Critical"])
        bn.add_node("UserFatigue", ["Low", "High"])
        bn.add_node(
            "RecommendedAction",
            [
                "BookCalendar",
                "SetImmediateAlarm",
                "ProvideConciseAnswer",
                "TriggerEmergencyProtocol",
                "PoliteGreeting",
                "AskClarification",
            ],
        )
        bn.add_node("RequireConfirmation", ["Yes", "No"])

        # 2. Add Directed Edges (Dependencies)
        bn.add_edge("TimeOfDay", "UserFatigue")
        bn.add_edge("UserSentiment", "UserFatigue")

        bn.add_edge("UserUrgency", "TaskImportance")
        bn.add_edge("DetectedIntent", "TaskImportance")

        bn.add_edge("TaskImportance", "RecommendedAction")
        bn.add_edge("UserFatigue", "RecommendedAction")
        bn.add_edge("DetectedIntent", "RecommendedAction")

        bn.add_edge("RecommendedAction", "RequireConfirmation")

        # 3. Define Prior CPTs
        # P(TimeOfDay)
        p_time = np.array([0.30, 0.35, 0.25, 0.10])
        bn.set_cpt("TimeOfDay", p_time)

        # P(UserUrgency)
        p_urgency = np.array([0.70, 0.25, 0.05])
        bn.set_cpt("UserUrgency", p_urgency)

        # P(UserSentiment)
        p_sentiment = np.array([0.65, 0.15, 0.20])
        bn.set_cpt("UserSentiment", p_sentiment)

        # P(DetectedIntent)
        p_intent = np.array([0.25, 0.25, 0.30, 0.05, 0.15])
        bn.set_cpt("DetectedIntent", p_intent)

        # 4. Conditional CPTs
        # P(UserFatigue | TimeOfDay, UserSentiment) -> shape (4, 3, 2)
        # Fatigue is higher at Night/Evening and when Frustrated/Busy
        cpt_fatigue = np.zeros((4, 3, 2))
        for t_idx, time in enumerate(["Morning", "Afternoon", "Evening", "Night"]):
            for s_idx, sent in enumerate(["Calm", "Frustrated", "Busy"]):
                base_high_prob = 0.15
                if time in ["Evening", "Night"]:
                    base_high_prob += 0.40
                if sent in ["Frustrated", "Busy"]:
                    base_high_prob += 0.35
                base_high_prob = min(max(base_high_prob, 0.05), 0.95)
                cpt_fatigue[t_idx, s_idx] = [1.0 - base_high_prob, base_high_prob]
        bn.set_cpt("UserFatigue", cpt_fatigue)

        # P(TaskImportance | UserUrgency, DetectedIntent) -> shape (3, 5, 3)
        # States: ["Normal", "High", "Critical"]
        cpt_importance = np.zeros((3, 5, 3))
        for u_idx, urg in enumerate(["Normal", "High", "Critical"]):
            for i_idx, intent in enumerate(["Schedule", "Reminder", "Query", "HealthCheck", "Greeting"]):
                if urg == "Critical" or intent == "HealthCheck":
                    probs = [0.05, 0.25, 0.70]
                elif urg == "High" or intent in ["Schedule", "Reminder"]:
                    probs = [0.15, 0.65, 0.20]
                elif intent == "Greeting":
                    probs = [0.85, 0.10, 0.05]
                else:
                    probs = [0.60, 0.30, 0.10]
                cpt_importance[u_idx, i_idx] = probs
        bn.set_cpt("TaskImportance", cpt_importance)

        # P(RecommendedAction | TaskImportance, UserFatigue, DetectedIntent)
        # shape (3, 2, 5, 6)
        # Actions: ["BookCalendar", "SetImmediateAlarm", "ProvideConciseAnswer", "TriggerEmergencyProtocol", "PoliteGreeting", "AskClarification"]
        cpt_action = np.zeros((3, 2, 5, 6))
        actions = ["BookCalendar", "SetImmediateAlarm", "ProvideConciseAnswer", "TriggerEmergencyProtocol", "PoliteGreeting", "AskClarification"]
        a_idx = {a: i for i, a in enumerate(actions)}

        for imp_idx, imp in enumerate(["Normal", "High", "Critical"]):
            for f_idx, fat in enumerate(["Low", "High"]):
                for int_idx, intent in enumerate(["Schedule", "Reminder", "Query", "HealthCheck", "Greeting"]):
                    dist = np.full(6, 0.02)
                    if imp == "Critical" or intent == "HealthCheck":
                        dist[a_idx["TriggerEmergencyProtocol"]] = 0.85
                        dist[a_idx["SetImmediateAlarm"]] = 0.09
                    elif intent == "Schedule":
                        dist[a_idx["BookCalendar"]] = 0.80
                        dist[a_idx["AskClarification"]] = 0.10
                    elif intent == "Reminder":
                        dist[a_idx["SetImmediateAlarm"]] = 0.80
                        dist[a_idx["AskClarification"]] = 0.10
                    elif intent == "Greeting":
                        dist[a_idx["PoliteGreeting"]] = 0.88
                    elif intent == "Query":
                        dist[a_idx["ProvideConciseAnswer"]] = 0.80
                        dist[a_idx["AskClarification"]] = 0.10

                    # If fatigue is high, bias slightly toward concise actions
                    if fat == "High":
                        dist[a_idx["ProvideConciseAnswer"]] += 0.05
                    dist = dist / np.sum(dist)
                    cpt_action[imp_idx, f_idx, int_idx] = dist

        bn.set_cpt("RecommendedAction", cpt_action)

        # P(RequireConfirmation | RecommendedAction) -> shape (6, 2)
        # States: ["Yes", "No"]
        cpt_confirm = np.zeros((6, 2))
        for act_idx, act in enumerate(actions):
            if act in ["BookCalendar", "TriggerEmergencyProtocol"]:
                cpt_confirm[act_idx] = [0.85, 0.15]
            elif act == "SetImmediateAlarm":
                cpt_confirm[act_idx] = [0.50, 0.50]
            else:
                cpt_confirm[act_idx] = [0.10, 0.90]
        bn.set_cpt("RequireConfirmation", cpt_confirm)

    def infer_action(
        self,
        detected_intent: str,
        user_urgency: str = "Normal",
        time_of_day: str = "Morning",
        user_sentiment: str = "Calm",
    ) -> Dict[str, Any]:
        """
        Execute Exact Variable Elimination given observed evidence.

        Returns:
            Dictionary with posterior distributions and top recommended action.
        """
        # Map detected intent name to BN state
        intent_mapping = {
            "INTENT_SCHEDULE": "Schedule",
            "INTENT_REMINDER": "Reminder",
            "INTENT_QUERY": "Query",
            "INTENT_HEALTH": "HealthCheck",
            "GREETING": "Greeting",
            "Schedule": "Schedule",
            "Reminder": "Reminder",
            "Query": "Query",
            "HealthCheck": "HealthCheck",
            "Greeting": "Greeting",
        }
        clean_intent = intent_mapping.get(detected_intent, "Query")

        evidence = {
            "DetectedIntent": clean_intent,
            "UserUrgency": user_urgency,
            "TimeOfDay": time_of_day,
            "UserSentiment": user_sentiment,
        }

        # Query RecommendedAction posterior
        action_posteriors = self.bn.variable_elimination(
            query_var="RecommendedAction",
            evidence=evidence,
        )

        # Query TaskImportance posterior
        importance_posteriors = self.bn.variable_elimination(
            query_var="TaskImportance",
            evidence=evidence,
        )

        # Query UserFatigue posterior
        fatigue_posteriors = self.bn.variable_elimination(
            query_var="UserFatigue",
            evidence=evidence,
        )

        # Determine best action
        best_action = max(action_posteriors.items(), key=lambda x: x[1])

        # Query RequireConfirmation given best action
        confirm_posteriors = self.bn.variable_elimination(
            query_var="RequireConfirmation",
            evidence={**evidence, "RecommendedAction": best_action[0]},
        )

        return {
            "evidence": evidence,
            "best_action": best_action[0],
            "best_action_confidence": float(best_action[1]),
            "action_posteriors": action_posteriors,
            "importance_posteriors": importance_posteriors,
            "fatigue_posteriors": fatigue_posteriors,
            "require_confirmation": confirm_posteriors.get("Yes", 0.0) > 0.5,
            "confirmation_probability": confirm_posteriors.get("Yes", 0.0),
        }
