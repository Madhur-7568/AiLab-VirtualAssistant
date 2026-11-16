"""
FastAPI Backend & Vercel Serverless Function Entry Point
-------------------------------------------------------
Exposes RESTful endpoints for:
- Virtual Assistant Conversational Pipeline (/api/chat)
- HMM Viterbi & Forward-Backward Decoding (/api/hmm/viterbi)
- Baum-Welch EM Unsupervised Training (/api/em/train)
- Bayesian Network Variable Elimination (/api/bayes/infer)
- Network Metadata & Topology (/api/network)
"""

import sys
import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.assistant.pipeline import ProbabilisticVirtualAssistant

app = FastAPI(
    title="Probabilistic Virtual Assistant API",
    description="Virtual Assistant powered by Hidden Markov Models, EM Algorithm, and Bayesian Networks.",
    version="1.0.0",
)

# Enable CORS for web deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = ProbabilisticVirtualAssistant()


class ChatRequest(BaseModel):
    utterance: str
    user_sentiment: Optional[str] = "Calm"
    time_of_day: Optional[str] = None
    override_urgency: Optional[str] = None


class HMMRequest(BaseModel):
    utterance: str


class EMRequest(BaseModel):
    corpus: Optional[List[str]] = None
    iterations: Optional[int] = 15


class BayesInferRequest(BaseModel):
    query_var: str
    evidence: Optional[Dict[str, str]] = None


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "Probabilistic Virtual Assistant",
        "version": "1.0.0",
        "models": ["Hidden Markov Model (HMM)", "Expectation-Maximization (EM)", "Bayesian Network"],
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.utterance.strip():
        raise HTTPException(status_code=400, detail="Utterance cannot be empty.")
    result = assistant.process_utterance(
        utterance=req.utterance,
        user_sentiment=req.user_sentiment or "Calm",
        time_of_day=req.time_of_day,
        override_urgency=req.override_urgency,
    )
    return result


@app.post("/api/hmm/viterbi")
def hmm_viterbi(req: HMMRequest):
    res = assistant.tagger.tag_utterance(req.utterance)
    return res


@app.post("/api/em/train")
def em_train(req: EMRequest):
    corpus = req.corpus if req.corpus else None
    iterations = req.iterations or 15
    res = assistant.train_unsupervised_em(corpus=corpus, iterations=iterations)
    return res


@app.post("/api/bayes/infer")
def bayes_infer(req: BayesInferRequest):
    try:
        posteriors = assistant.decision_engine.bn.variable_elimination(
            query_var=req.query_var,
            evidence=req.evidence or {},
        )
        return {
            "query_var": req.query_var,
            "evidence": req.evidence or {},
            "posterior_distribution": posteriors,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/network")
def get_network_topology():
    bn = assistant.decision_engine.bn
    return {
        "nodes": bn.nodes,
        "node_states": bn.node_states,
        "parents": bn.parents,
        "children": bn.children,
    }
