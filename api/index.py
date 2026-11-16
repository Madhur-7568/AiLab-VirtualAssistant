"""
HealthPulse AI: Healthcare Virtual Assistant API
Vercel Serverless Function & FastAPI Application
"""

import sys
import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.assistant.health_assistant import HealthAssistant

app = FastAPI(
    title="HealthPulse AI Assistant API",
    description="Healthcare Virtual Assistant with LLM Integration, Symptom Triage, and Vitals Tracker.",
    version="2.0.0",
)

# Enable CORS for web deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = HealthAssistant()


class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None


class TriageRequest(BaseModel):
    primary_symptom: str
    severity: Optional[int] = 4
    duration_days: Optional[int] = 1
    additional_symptoms: Optional[List[str]] = None
    has_chronic_illness: Optional[bool] = False
    age: Optional[int] = None


class VitalsRequest(BaseModel):
    systolic: Optional[int] = None
    diastolic: Optional[int] = None
    heart_rate: Optional[int] = None
    blood_glucose: Optional[float] = None
    glucose_type: Optional[str] = "fasting"
    spo2: Optional[int] = None


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "HealthPulse AI - Healthcare Virtual Assistant",
        "version": "2.0.0",
        "llm_support": ["Google Gemini 1.5 Flash", "OpenAI GPT-4o-mini", "Clinical Knowledge Engine (Fallback)"],
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    result = assistant.chat(user_message=req.message, api_key_override=req.api_key)
    return result


@app.post("/api/triage")
def triage_endpoint(req: TriageRequest):
    if not req.primary_symptom.strip():
        raise HTTPException(status_code=400, detail="Primary symptom is required.")
    return assistant.check_symptoms(
        symptom=req.primary_symptom,
        severity=req.severity or 4,
        duration=req.duration_days or 1,
        additional=req.additional_symptoms or [],
        chronic=req.has_chronic_illness or False,
        age=req.age,
    )


@app.post("/api/vitals")
def vitals_endpoint(req: VitalsRequest):
    return assistant.analyze_vitals(
        systolic=req.systolic,
        diastolic=req.diastolic,
        heart_rate=req.heart_rate,
        blood_glucose=req.blood_glucose,
        glucose_type=req.glucose_type or "fasting",
        spo2=req.spo2,
    )
