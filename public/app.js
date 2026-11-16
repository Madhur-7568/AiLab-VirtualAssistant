/**
 * HealthPulse AI - Frontend Application Logic
 * Supports Voice Input, Speech Synthesis, Triage Evaluation, Vitals Analysis, and Gemini API Configuration.
 */

// Tab Navigation
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

    btn.classList.add('active');
    const tabId = btn.getAttribute('data-tab');
    document.getElementById(tabId).classList.add('active');
  });
});

// API Key Management (LocalStorage)
let userGeminiApiKey = localStorage.getItem('HEALTHPULSE_GEMINI_KEY') || '';

const keyModal = document.getElementById('key-modal');
const openKeyModalBtn = document.getElementById('open-key-modal-btn');
const closeKeyModalBtn = document.getElementById('close-modal-btn');
const saveKeyBtn = document.getElementById('save-key-btn');
const clearKeyBtn = document.getElementById('clear-key-btn');
const geminiKeyInput = document.getElementById('gemini-key-input');
const keyBtnLabel = document.getElementById('key-btn-label');
const activeModelTag = document.getElementById('active-model-tag');

function updateKeyUI() {
  if (userGeminiApiKey) {
    keyBtnLabel.textContent = 'API Key: Connected (Gemini)';
    activeModelTag.textContent = 'Engine: Google Gemini 1.5 Flash';
  } else {
    keyBtnLabel.textContent = 'API Key: Default';
    activeModelTag.textContent = 'Engine: Clinical Knowledge Base';
  }
}
updateKeyUI();

openKeyModalBtn.addEventListener('click', () => {
  geminiKeyInput.value = userGeminiApiKey;
  keyModal.classList.add('active');
});

closeKeyModalBtn.addEventListener('click', () => {
  keyModal.classList.remove('active');
});

saveKeyBtn.addEventListener('click', () => {
  const val = geminiKeyInput.value.trim();
  userGeminiApiKey = val;
  if (val) {
    localStorage.setItem('HEALTHPULSE_GEMINI_KEY', val);
  } else {
    localStorage.removeItem('HEALTHPULSE_GEMINI_KEY');
  }
  updateKeyUI();
  keyModal.classList.remove('active');
});

clearKeyBtn.addEventListener('click', () => {
  userGeminiApiKey = '';
  localStorage.removeItem('HEALTHPULSE_GEMINI_KEY');
  geminiKeyInput.value = '';
  updateKeyUI();
  keyModal.classList.remove('active');
});

// Chatbot Flow
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatMessages = document.getElementById('chat-messages');
const clearChatBtn = document.getElementById('clear-chat-btn');
const voiceInputBtn = document.getElementById('voice-input-btn');

// Quick Doubt Chips
document.querySelectorAll('.doubt-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    chatInput.value = chip.getAttribute('data-msg');
    chatForm.dispatchEvent(new Event('submit'));
  });
});

clearChatBtn.addEventListener('click', () => {
  chatMessages.innerHTML = '';
});

// Voice Input (Web Speech Recognition)
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    voiceInputBtn.style.color = '#ef4444';
    chatInput.placeholder = 'Listening... Speak your healthcare question now.';
  };

  recognition.onresult = (e) => {
    const transcript = e.results[0][0].transcript;
    chatInput.value = transcript;
    voiceInputBtn.style.color = '';
    chatInput.placeholder = "Ask your health question or doubt...";
    chatForm.dispatchEvent(new Event('submit'));
  };

  recognition.onerror = () => {
    voiceInputBtn.style.color = '';
    chatInput.placeholder = "Ask your health question or doubt...";
  };

  recognition.onend = () => {
    voiceInputBtn.style.color = '';
    chatInput.placeholder = "Ask your health question or doubt...";
  };

  voiceInputBtn.addEventListener('click', () => {
    recognition.start();
  });
} else {
  voiceInputBtn.title = 'Speech recognition not supported in this browser';
  voiceInputBtn.style.opacity = '0.5';
}

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  // Append User Message
  appendMessage('user', text);
  chatInput.value = '';

  // Append Thinking Message
  const loadingEl = appendThinkingMessage();

  try {
    let result = null;

    // Try backend API
    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          api_key: userGeminiApiKey || null,
        }),
      });
      if (resp.ok) {
        result = await resp.json();
      }
    } catch (err) {
      console.warn('API connection failed, falling back to client-side clinical reasoning', err);
    }

    // Client-side fallback if backend is offline
    if (!result) {
      result = clientFallbackChat(text);
    }

    loadingEl.remove();
    appendMessage('assistant', result.response, result.model_used, result.is_emergency);

  } catch (error) {
    loadingEl.remove();
    appendMessage('assistant', 'Sorry, I encountered an issue processing your query. Please try again.');
  }
});

function formatMarkdown(text) {
  return text
    .replace(/^### (.*$)/gim, '<h4 style="margin:8px 0 4px; color:#2dd4bf;">$1</h4>')
    .replace(/^## (.*$)/gim, '<h3 style="margin:10px 0 6px; color:#38bdf8;">$1</h3>')
    .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/gim, '<em>$1</em>')
    .replace(/\n\n/gim, '<br><br>')
    .replace(/\n/gim, '<br>');
}

function appendMessage(sender, text, modelUsed, isEmergency) {
  const msgEl = document.createElement('div');
  msgEl.className = `message ${sender}-msg`;
  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  const avatar = sender === 'assistant'
    ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>`
    : 'YOU';

  const badge = modelUsed ? `<span style="font-size:0.7rem; color:#2dd4bf; margin-left:8px;">(${modelUsed})</span>` : '';
  const emergencyStyle = isEmergency ? 'border: 2px solid #ef4444; background: rgba(239, 68, 68, 0.1);' : '';

  msgEl.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body" style="${emergencyStyle}">
      <div class="msg-meta">
        <span class="sender-title">${sender === 'assistant' ? 'HealthPulse AI' : 'You'} ${badge}</span>
        <span class="msg-time">${timeStr}</span>
      </div>
      <div class="msg-text">${formatMarkdown(text)}</div>
    </div>
  `;
  chatMessages.appendChild(msgEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendThinkingMessage() {
  const msgEl = document.createElement('div');
  msgEl.className = 'message assistant-msg';
  msgEl.innerHTML = `
    <div class="msg-avatar"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg></div>
    <div class="msg-body">
      <span style="color:#94a3b8; font-style:italic;">HealthPulse AI is analyzing medical guidelines...</span>
    </div>
  `;
  chatMessages.appendChild(msgEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return msgEl;
}

// -------------------------------------------------------------
// Symptom Triage Form Logic
// -------------------------------------------------------------
const triageForm = document.getElementById('triage-form');
const triageSeverity = document.getElementById('triage-severity');
const severityVal = document.getElementById('severity-val');
const triageResultContent = document.getElementById('triage-result-content');

triageSeverity.addEventListener('input', () => {
  const v = parseInt(triageSeverity.value);
  let label = 'Mild';
  if (v >= 8) label = 'Severe / Acute';
  else if (v >= 5) label = 'Moderate';
  severityVal.textContent = `${v} - ${label}`;
});

triageForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const symptom = document.getElementById('triage-symptom').value.trim();
  const severity = parseInt(triageSeverity.value);
  const duration = parseInt(document.getElementById('triage-duration').value) || 1;
  const chronic = document.getElementById('triage-chronic').checked;
  const age = document.getElementById('triage-age').value ? parseInt(document.getElementById('triage-age').value) : null;

  let result = null;
  try {
    const resp = await fetch('/api/triage', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        primary_symptom: symptom,
        severity,
        duration_days: duration,
        has_chronic_illness: chronic,
        age,
      }),
    });
    if (resp.ok) result = await resp.json();
  } catch (err) {
    console.warn(err);
  }

  if (!result) {
    result = clientFallbackTriage(symptom, severity, duration, chronic);
  }

  renderTriageResult(result);
});

function renderTriageResult(res) {
  triageResultContent.className = 'triage-card';
  triageResultContent.innerHTML = `
    <div class="triage-badge-banner" style="background: ${res.color}22; color: ${res.color}; border: 1px solid ${res.color};">
      <span>●</span> ${res.badge}
    </div>
    <div class="triage-detail-box">
      <div><strong>Recommended Clinical Action:</strong><br>${res.recommended_action}</div>
      <div><strong>Target Timeframe:</strong> ${res.timeframe}</div>
      <div><strong>Clinical Rationale:</strong> ${res.clinical_rationale}</div>
    </div>
    <div style="font-size:0.78rem; color:#94a3b8;">
      Evaluated for: <em>"${res.input_summary.primary_symptom}"</em> (Severity: ${res.input_summary.severity}, Duration: ${res.input_summary.duration_days} days)
    </div>
  `;
}

// -------------------------------------------------------------
// Vitals Form Logic
// -------------------------------------------------------------
const vitalsForm = document.getElementById('vitals-form');
const vitalsContainer = document.getElementById('vitals-results-container');

vitalsForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const sys = document.getElementById('vital-systolic').value ? parseInt(document.getElementById('vital-systolic').value) : null;
  const dia = document.getElementById('vital-diastolic').value ? parseInt(document.getElementById('vital-diastolic').value) : null;
  const hr = document.getElementById('vital-hr').value ? parseInt(document.getElementById('vital-hr').value) : null;
  const spo2 = document.getElementById('vital-spo2').value ? parseInt(document.getElementById('vital-spo2').value) : null;
  const gluc = document.getElementById('vital-glucose').value ? parseFloat(document.getElementById('vital-glucose').value) : null;
  const glucType = document.getElementById('vital-glucose-type').value;

  let result = null;
  try {
    const resp = await fetch('/api/vitals', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        systolic: sys,
        diastolic: dia,
        heart_rate: hr,
        spo2,
        blood_glucose: gluc,
        glucose_type: glucType,
      }),
    });
    if (resp.ok) result = await resp.json();
  } catch (err) {
    console.warn(err);
  }

  if (!result) {
    result = clientFallbackVitals(sys, dia, hr, spo2, gluc, glucType);
  }

  renderVitalsResults(result);
});

function renderVitalsResults(data) {
  vitalsContainer.innerHTML = '';

  const cards = [
    { key: 'blood_pressure', title: 'Blood Pressure', unit: 'mmHg' },
    { key: 'heart_rate', title: 'Heart Rate', unit: 'BPM' },
    { key: 'blood_glucose', title: 'Blood Glucose', unit: 'mg/dL' },
    { key: 'spo2', title: 'Oxygen SpO2', unit: '%' },
  ];

  cards.forEach(c => {
    const item = data[c.key];
    const cardEl = document.createElement('div');
    cardEl.className = 'vital-card';

    if (item) {
      cardEl.innerHTML = `
        <h4>${c.title}</h4>
        <div class="vital-val" style="color: ${item.color}">${item.reading}</div>
        <div class="vital-cat" style="color: ${item.color}">${item.category}</div>
        <div class="vital-adv">${item.clinical_advice}</div>
      `;
    } else {
      cardEl.classList.add('empty-vital');
      cardEl.innerHTML = `<h4>${c.title}</h4><p>No measurement entered.</p>`;
    }
    vitalsContainer.appendChild(cardEl);
  });
}

// -------------------------------------------------------------
// Client Fallback Engines (Guarantees 100% Zero-Setup Offline Reliability)
// -------------------------------------------------------------
function clientFallbackChat(query) {
  const q = query.toLowerCase();

  // Emergency check
  if (q.includes('chest pain') || q.includes('cannot breathe') || q.includes('shortness of breath') || q.includes('unconscious') || q.includes('stroke')) {
    return {
      response: `🚨 **CRITICAL MEDICAL ALERT: IMMEDIATE ATTENTION REQUIRED**\n\nYou described symptoms that may indicate a serious medical emergency.\n\n### Immediate Actions:\n1. **Call Emergency Services:** Dial **911** or **112** immediately.\n2. **Do not drive yourself:** Request an ambulance for immediate medical transit.\n3. **Stay seated and calm:** Loosen restrictive clothing.\n\n*Seek immediate in-person emergency medical care.*`,
      model_used: 'Emergency Safety Protocol',
      is_emergency: true,
    };
  }

  if (q.includes('fever')) {
    return {
      response: `🩺 **Clinical Overview: Fever**\n\n### Self-Care Measures:\n- **Hydration:** Drink plenty of oral rehydration fluids or warm broths.\n- **Rest:** Allow your body time to recover.\n- **OTC Relief:** Paracetamol (Acetaminophen) or Ibuprofen (take with food).\n\n### When to Consult a Doctor:\n- Temperature > 103°F (39.4°C)\n- Stiff neck, rash, or lasting longer than 3 days.`,
      model_used: 'Clinical Knowledge Base',
      is_emergency: false,
    };
  }

  if (q.includes('paracetamol') && q.includes('ibuprofen')) {
    return {
      response: `💊 **Medication Consultation: Paracetamol & Ibuprofen Combination**\n\n- **Safety:** Yes, Paracetamol and Ibuprofen belong to different drug classes (analgesic vs NSAID) and can be safely alternated or taken together if indicated.\n- **Dosing Strategy:** Many doctors recommend staggering them every 3-4 hours (e.g. Paracetamol at 12:00, Ibuprofen at 15:00) to maintain consistent pain/fever relief.\n- **Crucial Rule:** Always take Ibuprofen with food to protect your stomach lining. Do not exceed 4000mg Paracetamol/day or 1200mg OTC Ibuprofen/day.\n\n*Consult your doctor if you have kidney, liver, or stomach ulcer conditions.*`,
      model_used: 'Clinical Knowledge Base',
      is_emergency: false,
    };
  }

  return {
    response: `🩺 **Healthcare Virtual Assistant Guidance**\n\nThank you for reaching out with your question: *"${query}"*\n\n### General Health Recommendations:\n1. **Monitor Symptoms:** Track the onset, severity, and any aggravating factors.\n2. **Hydration & Rest:** Ensure consistent fluid intake and 7-8 hours of sleep.\n3. **When to seek medical attention:** If pain worsens, lasts > 48 hours, or is accompanied by fever or breathing changes.\n\n💬 *You can also use the Symptom Triage tab above for a structured urgency score!*`,
    model_used: 'Clinical Knowledge Base',
    is_emergency: false,
  };
}

function clientFallbackTriage(symptom, severity, duration, chronic) {
  let level = 'SELF_CARE';
  let badge = 'SELF-CARE';
  let color = '#10b981';
  let action = 'Rest, maintain hydration, and monitor at home.';
  let timeframe = '48 - 72 hours';
  let rationale = 'Mild symptom presentation suitable for home management.';

  const s = symptom.toLowerCase();
  if (s.includes('chest pain') || s.includes('breath') || severity >= 9) {
    level = 'EMERGENCY';
    badge = 'EMERGENCY (Immediate 911 / 112)';
    color = '#ef4444';
    action = 'Go to nearest Emergency Department or call 911/112 immediately.';
    timeframe = 'Immediate (minutes)';
    rationale = 'High acute severity or critical red-flag symptom detected.';
  } else if (severity >= 7 || (duration > 7 && severity >= 5)) {
    level = 'URGENT_CARE';
    badge = 'URGENT CARE';
    color = '#f97316';
    action = 'Visit an Urgent Care clinic or same-day clinic consultation.';
    timeframe = 'Within 12 to 24 hours';
    rationale = `High severity score (${severity}/10) or prolonged duration.`;
  } else if (duration >= 3 || severity >= 4 || chronic) {
    level = 'ROUTINE_CLINIC';
    badge = 'ROUTINE CLINIC';
    color = '#eab308';
    action = 'Schedule an appointment with your primary care physician.';
    timeframe = 'Within 2 to 3 days';
    rationale = 'Persistent moderate symptoms warranting medical evaluation.';
  }

  return {
    triage_level: level,
    badge,
    color,
    recommended_action: action,
    timeframe,
    clinical_rationale: rationale,
    input_summary: {
      primary_symptom: symptom,
      severity: `${severity}/10`,
      duration_days: duration,
    },
  };
}

function clientFallbackVitals(sys, dia, hr, spo2, gluc, glucType) {
  const res = {};
  if (sys && dia) {
    let cat = 'Normal Blood Pressure';
    let col = '#10b981';
    let adv = 'Optimal cardiovascular health.';
    if (sys > 180 || dia > 120) {
      cat = 'Hypertensive Crisis (EMERGENCY)';
      col = '#ef4444';
      adv = 'Seek immediate emergency medical attention.';
    } else if (sys >= 140 || dia >= 90) {
      cat = 'Stage 2 Hypertension';
      col = '#f97316';
      adv = 'Consult your physician for prescription treatment.';
    } else if (sys >= 130 || dia >= 80) {
      cat = 'Stage 1 Hypertension';
      col = '#eab308';
      adv = 'Adopt low-sodium diet and regular aerobic exercise.';
    }
    res.blood_pressure = { reading: `${sys}/${dia} mmHg`, category: cat, color: col, clinical_advice: adv };
  }

  if (hr) {
    let cat = 'Normal Resting Heart Rate';
    let col = '#10b981';
    let adv = 'Heart rate is within optimal resting range (60-100 BPM).';
    if (hr > 100) { cat = 'Tachycardia (Elevated HR)'; col = '#f97316'; adv = 'Rest, hydrate, and recheck.'; }
    else if (hr < 60) { cat = 'Bradycardia (Low HR)'; col = '#38bdf8'; adv = 'Normal in athletes; seek review if dizzy.'; }
    res.heart_rate = { reading: `${hr} BPM`, category: cat, color: col, clinical_advice: adv };
  }

  if (spo2) {
    let cat = 'Normal Oxygen Saturation';
    let col = '#10b981';
    let adv = 'Healthy blood oxygen level (95-100%).';
    if (spo2 < 90) { cat = 'Critical Hypoxemia'; col = '#ef4444'; adv = 'Seek emergency medical attention immediately.'; }
    else if (spo2 <= 94) { cat = 'Mild Hypoxia'; col = '#f97316'; adv = 'Lower than normal; medical review advised.'; }
    res.spo2 = { reading: `${spo2}%`, category: cat, color: col, clinical_advice: adv };
  }

  if (gluc) {
    let cat = 'Normal Fasting Glucose';
    let col = '#10b981';
    let adv = 'Healthy glucose metabolism.';
    if (gluc < 70) { cat = 'Hypoglycemia'; col = '#ef4444'; adv = 'Take 15g fast-acting sugar and retest.'; }
    else if (gluc >= 126) { cat = 'Diabetes Range (Fasting)'; col = '#f97316'; adv = 'Consult doctor for confirmation HbA1c test.'; }
    else if (gluc >= 100) { cat = 'Pre-diabetes'; col = '#eab308'; adv = 'Lifestyle diet changes can reverse progression.'; }
    res.blood_glucose = { reading: `${gluc} mg/dL`, category: cat, color: col, clinical_advice: adv };
  }

  return res;
}
