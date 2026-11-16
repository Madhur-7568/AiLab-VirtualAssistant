"""
Integration tests for the Virtual Assistant Pipeline
"""

from src.assistant.pipeline import ProbabilisticVirtualAssistant


def test_virtual_assistant_greeting():
    bot = ProbabilisticVirtualAssistant()
    res = bot.process_utterance("Hello good morning assistant")

    assert res["hmm"]["primary_intent"] == "GREETING"
    assert res["bayesian_network"]["selected_action"] == "PoliteGreeting"
    assert "Hello" in res["response"]


def test_virtual_assistant_schedule_task():
    bot = ProbabilisticVirtualAssistant()
    res = bot.process_utterance("Schedule urgent meeting tomorrow morning")

    assert res["hmm"]["primary_intent"] == "INTENT_SCHEDULE"
    assert res["hmm"]["slots"]["urgency"] == "High"
    assert "BookCalendar" in res["bayesian_network"]["selected_action"]
    assert "meeting" in res["response"].lower()


def test_virtual_assistant_emergency_health():
    bot = ProbabilisticVirtualAssistant()
    res = bot.process_utterance("Emergency call the doctor right now")

    assert res["hmm"]["primary_intent"] == "INTENT_HEALTH"
    assert res["bayesian_network"]["selected_action"] == "TriggerEmergencyProtocol"
    assert "EMERGENCY" in res["response"]


def test_virtual_assistant_unsupervised_training():
    bot = ProbabilisticVirtualAssistant()
    training_res = bot.train_unsupervised_em(iterations=5)

    assert training_res["num_utterances"] > 0
    assert len(training_res["log_likelihood_curve"]) > 0
