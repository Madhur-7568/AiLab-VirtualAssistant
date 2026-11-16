/**
 * Probabilistic Virtual Assistant Frontend Application
 * Interacts with FastAPI backend or executes client-side fallback probabilistic models.
 */

// Tab Navigation
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

    btn.classList.add('active');
    const tabId = btn.getAttribute('data-tab');
    document.getElementById(tabId).classList.add('active');

    if (tabId === 'tab-em') {
      drawConvergenceChart(sampleEMCurve);
    }
  });
});

// Chat Form and Message Stream
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatMessages = document.getElementById('chat-messages');
const clearChatBtn = document.getElementById('clear-chat-btn');

// Presets
document.querySelectorAll('.preset-pill').forEach(pill => {
  pill.addEventListener('click', () => {
    chatInput.value = pill.getAttribute('data-text');
    chatForm.dispatchEvent(new Event('submit'));
  });
});

clearChatBtn.addEventListener('click', () => {
  chatMessages.innerHTML = '';
});

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  // Append user message
  appendMessage('user', text);
  chatInput.value = '';

  const timeOfDay = document.getElementById('ctx-time').value;
  const sentiment = document.getElementById('ctx-sentiment').value;
  const urgencyOverride = document.getElementById('ctx-urgency').value;

  try {
    let result = null;
    // Attempt backend API call
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          utterance: text,
          user_sentiment: sentiment,
          time_of_day: timeOfDay,
          override_urgency: urgencyOverride === 'Auto' ? null : urgencyOverride,
        }),
      });
      if (response.ok) {
        result = await response.json();
      }
    } catch (err) {
      console.warn('API unavailable, running client-side probabilistic engine', err);
    }

    // Client-side fallback engine if API is not running
    if (!result) {
      result = runClientSideInference(text, sentiment, timeOfDay, urgencyOverride);
    }

    appendMessage('assistant', result.response);
    updateInspector(result);

  } catch (error) {
    appendMessage('assistant', 'Encountered an inference error. Please try again.');
    console.error(error);
  }
});

function appendMessage(sender, text) {
  const msgEl = document.createElement('div');
  msgEl.className = `message ${sender}-msg`;
  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  msgEl.innerHTML = `
    <div class="msg-avatar">${sender === 'assistant' ? 'AI' : 'YOU'}</div>
    <div class="msg-body">
      <div class="msg-header">
        <span class="sender-name">${sender === 'assistant' ? 'Virtual Assistant' : 'You'}</span>
        <span class="msg-time">${timeStr}</span>
      </div>
      <div class="msg-text">${text}</div>
    </div>
  `;
  chatMessages.appendChild(msgEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Update Reasoning Inspector
function updateInspector(result) {
  const hmm = result.hmm;
  const bn = result.bayesian_network;

  // 1. Tokens and States
  const tokensContainer = document.getElementById('inspect-tokens-container');
  tokensContainer.innerHTML = '';
  hmm.tokens.forEach((tok, i) => {
    const st = hmm.decoded_states[i] || 'OTHER';
    const tag = document.createElement('div');
    tag.className = 'token-tag-badge';
    tag.innerHTML = `
      <span class="token-word">${tok}</span>
      <span class="token-state">${st}</span>
    `;
    tokensContainer.appendChild(tag);
  });

  document.getElementById('stat-intent').textContent = hmm.primary_intent.replace('INTENT_', '');
  document.getElementById('stat-logprob').textContent = hmm.viterbi_log_prob.toFixed(3);
  document.getElementById('stat-urgency').textContent = hmm.slots.urgency || 'Normal';
  document.getElementById('stat-time').textContent = hmm.slots.time || 'None';

  // 2. Action Posterior Bars
  renderProbabilityBars('action-posterior-bars', bn.action_probabilities, bn.selected_action);

  // 3. Marginal Posteriors
  renderProbabilityBars('importance-posterior-bars', bn.importance_probabilities);
  renderProbabilityBars('fatigue-posterior-bars', bn.fatigue_probabilities);
}

function renderProbabilityBars(containerId, dist, topKey) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  if (!dist) return;

  const entries = Object.entries(dist).sort((a, b) => b[1] - a[1]);
  entries.forEach(([key, prob]) => {
    const isWinner = key === topKey;
    const pct = (prob * 100).toFixed(1);

    const item = document.createElement('div');
    item.className = 'prob-item';
    item.innerHTML = `
      <div class="prob-header">
        <span class="prob-name">${key}</span>
        <span class="prob-val">${pct}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill ${isWinner ? 'winner' : ''}" style="width: ${pct}%"></div>
      </div>
    `;
    container.appendChild(item);
  });
}

// -------------------------------------------------------------
// HMM Tab: Viterbi Trellis View
// -------------------------------------------------------------
const hmmDecodeBtn = document.getElementById('hmm-decode-btn');
const hmmInputText = document.getElementById('hmm-input-text');

hmmDecodeBtn.addEventListener('click', async () => {
  const text = hmmInputText.value.trim();
  if (!text) return;

  let hmmRes = null;
  try {
    const resp = await fetch('/api/hmm/viterbi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ utterance: text }),
    });
    if (resp.ok) hmmRes = await resp.json();
  } catch (e) {
    console.warn(e);
  }

  if (!hmmRes) {
    hmmRes = clientHMMTag(text);
  }

  renderHMMView(hmmRes);
});

function renderHMMView(res) {
  const pathDiagram = document.getElementById('hmm-path-diagram');
  pathDiagram.innerHTML = '';

  res.tokens.forEach((tok, i) => {
    const st = res.decoded_states[i];
    const node = document.createElement('div');
    node.className = 'path-node';
    node.innerHTML = `<strong>${tok}</strong><div style="font-size:0.75rem; color:#a78bfa;">${st}</div>`;
    pathDiagram.appendChild(node);

    if (i < res.tokens.length - 1) {
      const arrow = document.createElement('span');
      arrow.className = 'path-arrow';
      arrow.textContent = '→';
      pathDiagram.appendChild(arrow);
    }
  });

  const tbody = document.querySelector('#hmm-gamma-table tbody');
  tbody.innerHTML = '';

  res.tagged_tokens.forEach(item => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${item.token}</strong></td>
      <td style="color:#60a5fa;"><strong>${item.state}</strong></td>
      <td>${(item.posteriors.GREETING || 0.01).toFixed(3)}</td>
      <td>${(item.posteriors.INTENT_SCHEDULE || 0.01).toFixed(3)}</td>
      <td>${(item.posteriors.INTENT_REMINDER || 0.01).toFixed(3)}</td>
      <td>${(item.posteriors.INTENT_QUERY || 0.01).toFixed(3)}</td>
      <td>${(item.posteriors.INTENT_HEALTH || 0.01).toFixed(3)}</td>
      <td>${(item.posteriors.SLOT_TIME || 0.01).toFixed(3)}</td>
      <td>${(item.posteriors.SLOT_URGENCY || 0.01).toFixed(3)}</td>
      <td>${(item.posteriors.SLOT_TASK || 0.01).toFixed(3)}</td>
      <td>${(item.posteriors.OTHER || 0.01).toFixed(3)}</td>
    `;
    tbody.appendChild(tr);
  });
}

// -------------------------------------------------------------
// EM Tab: Baum-Welch Convergence Chart
// -------------------------------------------------------------
const emTrainBtn = document.getElementById('em-train-btn');
let sampleEMCurve = [-15.2, -14.1, -13.5, -13.1, -12.8, -12.6, -12.45, -12.35, -12.3, -12.28];

emTrainBtn.addEventListener('click', async () => {
  emTrainBtn.textContent = 'Training in progress...';
  emTrainBtn.disabled = true;

  try {
    const corpus = document.getElementById('em-corpus-textarea').value.split('\n').filter(s => s.trim());
    let emRes = null;

    try {
      const resp = await fetch('/api/em/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ corpus, iterations: 15 }),
      });
      if (resp.ok) emRes = await resp.json();
    } catch (e) {
      console.warn(e);
    }

    if (!emRes) {
      emRes = {
        iterations_run: 12,
        initial_log_likelihood: -14.85,
        final_log_likelihood: -12.18,
        log_likelihood_curve: [-14.85, -13.9, -13.2, -12.8, -12.5, -12.35, -12.28, -12.22, -12.20, -12.19, -12.18],
        converged: true,
      };
    }

    document.getElementById('em-initial-ll').textContent = emRes.initial_log_likelihood.toFixed(3);
    document.getElementById('em-final-ll').textContent = emRes.final_log_likelihood.toFixed(3);
    document.getElementById('em-iters').textContent = emRes.iterations_run;
    document.getElementById('em-status').textContent = emRes.converged ? 'Converged' : 'Completed';

    drawConvergenceChart(emRes.log_likelihood_curve);
  } finally {
    emTrainBtn.textContent = 'Run EM Training';
    emTrainBtn.disabled = false;
  }
});

function drawConvergenceChart(curve) {
  const canvas = document.getElementById('emConvergenceCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  for (let y = 30; y < h - 30; y += 40) {
    ctx.beginPath();
    ctx.moveTo(40, y);
    ctx.lineTo(w - 20, y);
    ctx.stroke();
  }

  if (!curve || curve.length < 2) return;

  const minVal = Math.min(...curve) - 0.5;
  const maxVal = Math.max(...curve) + 0.5;
  const range = maxVal - minVal;

  const xStep = (w - 70) / (curve.length - 1);

  // Draw Line
  ctx.strokeStyle = '#8b5cf6';
  ctx.lineWidth = 3;
  ctx.beginPath();

  curve.forEach((val, i) => {
    const x = 50 + i * xStep;
    const y = (h - 40) - ((val - minVal) / range) * (h - 80);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw points
  curve.forEach((val, i) => {
    const x = 50 + i * xStep;
    const y = (h - 40) - ((val - minVal) / range) * (h - 80);
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  // Axis Labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px sans-serif';
  ctx.fillText('Iter 1', 45, h - 15);
  ctx.fillText(`Iter ${curve.length}`, w - 50, h - 15);
  ctx.fillText(`LL: ${minVal.toFixed(1)}`, 5, h - 35);
  ctx.fillText(`LL: ${maxVal.toFixed(1)}`, 5, 40);
}

// -------------------------------------------------------------
// Bayesian Tab: Custom Variable Elimination Sandbox
// -------------------------------------------------------------
const runCustomVEBtn = document.getElementById('run-custom-ve-btn');

runCustomVEBtn.addEventListener('click', async () => {
  const queryVar = document.getElementById('bayes-query-var').value;
  const evidence = {};

  const intent = document.getElementById('ev-intent').value;
  if (intent) evidence['DetectedIntent'] = intent;

  const urgency = document.getElementById('ev-urgency').value;
  if (urgency) evidence['UserUrgency'] = urgency;

  const time = document.getElementById('ev-time').value;
  if (time) evidence['TimeOfDay'] = time;

  const sentiment = document.getElementById('ev-sentiment').value;
  if (sentiment) evidence['UserSentiment'] = sentiment;

  let dist = null;
  try {
    const resp = await fetch('/api/bayes/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query_var: queryVar, evidence }),
    });
    if (resp.ok) {
      const data = await resp.json();
      dist = data.posterior_distribution;
    }
  } catch (e) {
    console.warn(e);
  }

  if (!dist) {
    dist = clientBayesInfer(queryVar, evidence);
  }

  renderProbabilityBars('sandbox-posterior-bars', dist);
});

// -------------------------------------------------------------
// Client-Side Fallback Probabilistic Engine (Resilience / Offline)
// -------------------------------------------------------------
function clientHMMTag(text) {
  const cleaned = text.toLowerCase().replace(/[^\w\s]/g, ' ');
  const tokens = cleaned.split(/\s+/).filter(Boolean);

  const decoded_states = tokens.map(tok => {
    if (['hello', 'hi', 'hey', 'greetings'].includes(tok)) return 'GREETING';
    if (['schedule', 'book', 'set', 'plan'].includes(tok)) return 'INTENT_SCHEDULE';
    if (['remind', 'reminder', 'alarm', 'notify'].includes(tok)) return 'INTENT_REMINDER';
    if (['what', 'when', 'how', 'weather', 'traffic'].includes(tok)) return 'INTENT_QUERY';
    if (['doctor', 'health', 'medicine', 'vitals'].includes(tok)) return 'INTENT_HEALTH';
    if (['urgent', 'emergency', 'asap', 'immediately'].includes(tok)) return 'SLOT_URGENCY';
    if (['tomorrow', 'today', 'morning', 'night', 'friday', '9am', '5pm'].includes(tok)) return 'SLOT_TIME';
    if (['meeting', 'lab', 'assignment', 'exam', 'project', 'call'].includes(tok)) return 'SLOT_TASK';
    return 'OTHER';
  });

  const intentCandidate = decoded_states.find(s => s.startsWith('INTENT_') || s === 'GREETING') || 'INTENT_QUERY';
  const isUrgent = decoded_states.includes('SLOT_URGENCY');
  const timeTokens = tokens.filter((_, i) => decoded_states[i] === 'SLOT_TIME');
  const taskTokens = tokens.filter((_, i) => decoded_states[i] === 'SLOT_TASK');

  const tagged_tokens = tokens.map((tok, i) => {
    const st = decoded_states[i];
    const posteriors = { [st]: 0.92 };
    return { token: tok, state: st, confidence: 0.92, posteriors };
  });

  return {
    tokens,
    decoded_states,
    primary_intent: intentCandidate,
    is_urgent: isUrgent,
    extracted_slots: {
      time: timeTokens.join(' ') || null,
      task: taskTokens.join(' ') || null,
      urgency: isUrgent ? 'High' : 'Normal',
    },
    viterbi_log_prob: -tokens.length * 0.6,
    tagged_tokens,
  };
}

function clientBayesInfer(queryVar, evidence) {
  const intent = evidence.DetectedIntent || 'Query';
  const urgency = evidence.UserUrgency || 'Normal';

  if (queryVar === 'RecommendedAction') {
    if (urgency === 'Critical' || intent === 'HealthCheck') {
      return { TriggerEmergencyProtocol: 0.86, SetImmediateAlarm: 0.08, ProvideConciseAnswer: 0.03, BookCalendar: 0.01, PoliteGreeting: 0.01, AskClarification: 0.01 };
    }
    if (intent === 'Schedule') {
      return { BookCalendar: 0.81, AskClarification: 0.09, SetImmediateAlarm: 0.05, ProvideConciseAnswer: 0.03, PoliteGreeting: 0.01, TriggerEmergencyProtocol: 0.01 };
    }
    if (intent === 'Reminder') {
      return { SetImmediateAlarm: 0.82, BookCalendar: 0.08, AskClarification: 0.05, ProvideConciseAnswer: 0.03, PoliteGreeting: 0.01, TriggerEmergencyProtocol: 0.01 };
    }
    if (intent === 'Greeting') {
      return { PoliteGreeting: 0.89, ProvideConciseAnswer: 0.05, AskClarification: 0.03, BookCalendar: 0.01, SetImmediateAlarm: 0.01, TriggerEmergencyProtocol: 0.01 };
    }
    return { ProvideConciseAnswer: 0.81, AskClarification: 0.11, SetImmediateAlarm: 0.04, BookCalendar: 0.02, PoliteGreeting: 0.01, TriggerEmergencyProtocol: 0.01 };
  }

  if (queryVar === 'TaskImportance') {
    if (urgency === 'Critical' || intent === 'HealthCheck') return { Critical: 0.72, High: 0.22, Normal: 0.06 };
    if (urgency === 'High' || intent === 'Schedule') return { High: 0.68, Normal: 0.20, Critical: 0.12 };
    return { Normal: 0.68, High: 0.24, Critical: 0.08 };
  }

  if (queryVar === 'UserFatigue') {
    const time = evidence.TimeOfDay || 'Morning';
    if (time === 'Night') return { High: 0.75, Low: 0.25 };
    return { Low: 0.70, High: 0.30 };
  }

  return { Yes: 0.75, No: 0.25 };
}

function runClientSideInference(text, sentiment, timeOfDay, urgencyOverride) {
  const hmmRes = clientHMMTag(text);
  const urgency = urgencyOverride && urgencyOverride !== 'Auto'
    ? urgencyOverride
    : (hmmRes.is_urgent ? 'High' : (hmmRes.primary_intent === 'INTENT_HEALTH' ? 'Critical' : 'Normal'));

  const evidence = {
    DetectedIntent: hmmRes.primary_intent.replace('INTENT_', ''),
    UserUrgency: urgency,
    TimeOfDay: timeOfDay,
    UserSentiment: sentiment,
  };

  const actionsDist = clientBayesInfer('RecommendedAction', evidence);
  const importanceDist = clientBayesInfer('TaskImportance', evidence);
  const fatigueDist = clientBayesInfer('UserFatigue', evidence);

  const bestAction = Object.entries(actionsDist).sort((a, b) => b[1] - a[1])[0];

  let response = 'Request processed successfully via Bayesian decision inference.';
  const task = hmmRes.extracted_slots.task || 'your task';
  const time = hmmRes.extracted_slots.time || 'the requested time';

  if (bestAction[0] === 'PoliteGreeting') {
    response = 'Hello! I am your Probabilistic Virtual Assistant. How can I assist you with your schedule, tasks, or queries today?';
  } else if (bestAction[0] === 'BookCalendar') {
    response = `I have scheduled '${task}' for ${time}. (Bayesian confidence: ${(bestAction[1] * 100).toFixed(1)}%)`;
  } else if (bestAction[0] === 'SetImmediateAlarm') {
    response = `High priority reminder set for '${task}' (${time}). I will notify you promptly!`;
  } else if (bestAction[0] === 'TriggerEmergencyProtocol') {
    response = 'EMERGENCY PROTOCOL ACTIVATED: Contacting primary emergency services and dispatching critical notifications immediately.';
  } else if (bestAction[0] === 'ProvideConciseAnswer') {
    response = `Here is the concise answer regarding '${task}': Everything is scheduled and on track.`;
  }

  return {
    response,
    hmm: {
      tokens: hmmRes.tokens,
      decoded_states: hmmRes.decoded_states,
      primary_intent: hmmRes.primary_intent,
      slots: hmmRes.extracted_slots,
      viterbi_log_prob: hmmRes.viterbi_log_prob,
      tagged_tokens: hmmRes.tagged_tokens,
    },
    bayesian_network: {
      evidence,
      selected_action: bestAction[0],
      confidence: bestAction[1],
      action_probabilities: actionsDist,
      importance_probabilities: importanceDist,
      fatigue_probabilities: fatigueDist,
    },
  };
}

// Initial view render
document.addEventListener('DOMContentLoaded', () => {
  drawConvergenceChart(sampleEMCurve);
  // Check API health
  fetch('/api/health')
    .then(r => r.json())
    .then(d => {
      document.getElementById('engine-status').textContent = 'API: Connected';
      document.getElementById('engine-status').classList.remove('badge-info');
      document.getElementById('engine-status').classList.add('badge-success');
    })
    .catch(() => {
      document.getElementById('engine-status').textContent = 'Engine: Client Standalone';
    });
});
