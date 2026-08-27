import { AutoTokenizer, AutoModelForSequenceClassification, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3';

// Configure environment for browser execution
env.allowLocalModels = false;

const MODEL_ID = 'Manjunath2000006/halluciguard-detector';

// DOM Elements
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const statusPercent = document.getElementById('statusPercent');
const progressBar = document.getElementById('progressBar');
const statusDetail = document.getElementById('statusDetail');

const detectorForm = document.getElementById('detectorForm');
const queryInput = document.getElementById('queryInput');
const responseInput = document.getElementById('responseInput');
const contextInput = document.getElementById('contextInput');
const contextToggle = document.getElementById('contextToggle');
const contextWrapper = document.getElementById('contextWrapper');
const contextArrow = document.getElementById('contextArrow');

const submitBtn = document.getElementById('submitBtn');
const btnSpinner = document.getElementById('btnSpinner');
const btnText = document.getElementById('btnText');

const presetTest1 = document.getElementById('presetTest1');
const presetTest2 = document.getElementById('presetTest2');
const presetClear = document.getElementById('presetClear');

const resultsCard = document.getElementById('resultsCard');
const resultBanner = document.getElementById('resultBanner');
const riskTag = document.getElementById('riskTag');
const bannerSummary = document.getElementById('bannerSummary');
const actionPill = document.getElementById('actionPill');
const actionText = document.getElementById('actionText');

const valHalProb = document.getElementById('valHalProb');
const subHalProb = document.getElementById('subHalProb');
const valConfidence = document.getElementById('valConfidence');
const valLabel = document.getElementById('valLabel');
const valLatency = document.getElementById('valLatency');

const diagToggle = document.getElementById('diagToggle');
const diagTable = document.getElementById('diagTable');
const diagArrow = document.getElementById('diagArrow');
const diagModelSource = document.getElementById('diagModelSource');
const diagLoaded = document.getElementById('diagLoaded');
const diagExecuted = document.getElementById('diagExecuted');
const diagDegraded = document.getElementById('diagDegraded');
const diagLogits = document.getElementById('diagLogits');
const diagProbs = document.getElementById('diagProbs');
const diagInput = document.getElementById('diagInput');

// Singleton model references
let tokenizerInstance = null;
let modelInstance = null;
let loadPromise = null;

// Progress tracker for multi-file download
const fileProgress = {};

function handleDownloadProgress(data) {
  if (data.status === 'progress') {
    fileProgress[data.file] = {
      loaded: data.loaded || 0,
      total: data.total || 0,
      progress: data.progress || 0
    };

    let totalLoaded = 0;
    let totalSize = 0;
    for (const f in fileProgress) {
      totalLoaded += fileProgress[f].loaded;
      totalSize += fileProgress[f].total;
    }

    const pct = totalSize > 0 ? Math.min(99, Math.round((totalLoaded / totalSize) * 100)) : 0;
    progressBar.style.width = `${pct}%`;
    statusPercent.textContent = `${pct}%`;
    statusText.textContent = `Downloading ${data.file}...`;
    statusDetail.textContent = `${(totalLoaded / (1024 * 1024)).toFixed(1)} MB downloaded`;
  } else if (data.status === 'initiate') {
    statusText.textContent = `Loading ${data.file}...`;
  } else if (data.status === 'done') {
    statusText.textContent = `Downloaded ${data.file}`;
  } else if (data.status === 'ready') {
    markModelReady();
  }
}

function markModelReady() {
  statusDot.classList.add('ready');
  statusText.textContent = 'Detector model loaded & ready in browser';
  statusPercent.textContent = '100%';
  progressBar.style.width = '100%';
  statusDetail.textContent = 'Singleton cached in browser storage (IndexedDB)';
}

/**
 * Singleton Model Loader
 * Loads tokenizer and ONNX sequence classification weights once, then reuses the cached instance.
 */
async function getDetectorPipeline() {
  if (tokenizerInstance && modelInstance) {
    return { tokenizer: tokenizerInstance, model: modelInstance };
  }

  if (!loadPromise) {
    loadPromise = (async () => {
      statusText.textContent = 'Downloading fine-tuned detector weights...';
      const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
        progress_callback: handleDownloadProgress
      });
      const model = await AutoModelForSequenceClassification.from_pretrained(MODEL_ID, {
        dtype: 'fp32',
        progress_callback: handleDownloadProgress
      });

      tokenizerInstance = tokenizer;
      modelInstance = model;
      markModelReady();
      return { tokenizer, model };
    })();
  }

  return loadPromise;
}

/**
 * Exact input formatter matching Python detector:
 * format_detector_input(query, response, context)
 */
function formatDetectorInput(query, response, context = null) {
  const q = (query || '').trim();
  const r = (response || '').trim();
  const c = (context || '').trim();
  if (c.length > 0) {
    return `Context: ${c}\nQuery: ${q}\nAnswer: ${r}`;
  }
  return `Query: ${q}\nAnswer: ${r}`;
}

/**
 * Numerically stable Softmax
 */
function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

/**
 * Threshold routing and metrics calculation
 * LOW <= 0.30 -> Accept
 * MEDIUM > 0.30 and < 0.50 -> Accept
 * HIGH >= 0.50 -> Verify
 */
function evaluateMetrics(probs) {
  const halProb = Number(probs[1].toFixed(4));
  const confScore = Number((1.0 - probs[1]).toFixed(4));

  let riskLevel = 'LOW';
  let nextAction = 'Accept';

  if (halProb >= 0.50) {
    riskLevel = 'HIGH';
    nextAction = 'Verify';
  } else if (halProb > 0.30) {
    riskLevel = 'MEDIUM';
    nextAction = 'Accept';
  } else {
    riskLevel = 'LOW';
    nextAction = 'Accept';
  }

  const predLabel = halProb >= 0.50 ? 1 : 0;
  const predName = predLabel === 1 ? 'HALLUCINATION' : 'NO_HALLUCINATION';

  return {
    hallucination_probability: halProb,
    confidence_score: confScore,
    predicted_label: predLabel,
    predicted_label_name: predName,
    risk_level: riskLevel,
    next_action: nextAction
  };
}

/**
 * Form Submit Handler
 */
async function handleDetection(e) {
  e.preventDefault();

  const query = queryInput.value.trim();
  const response = responseInput.value.trim();
  const context = contextInput.value.trim();

  if (!query || !response) {
    alert('Please provide both Query and Response.');
    return;
  }

  // Set loading state
  submitBtn.disabled = true;
  btnSpinner.style.display = 'inline-block';
  btnText.textContent = 'Analyzing...';

  const tStart = performance.now();

  try {
    const { tokenizer, model } = await getDetectorPipeline();

    const formattedInput = formatDetectorInput(query, response, context);
    const inputs = await tokenizer(formattedInput, {
      truncation: true,
      max_length: 384,
      padding: true
    });

    const output = await model(inputs);
    const rawLogits = Array.from(output.logits.data);
    const probs = softmax(rawLogits);
    const metrics = evaluateMetrics(probs);

    const latencyMs = Math.round(performance.now() - tStart);

    renderResults(metrics, rawLogits, probs, formattedInput, latencyMs);
  } catch (err) {
    console.error('Detection failed:', err);
    alert('Detection error: ' + err.message);
  } finally {
    submitBtn.disabled = false;
    btnSpinner.style.display = 'none';
    btnText.textContent = 'Run Hallucination Detection';
  }
}

/**
 * Render Results to UI
 */
function renderResults(metrics, logits, probs, formattedInput, latencyMs) {
  // Update banner
  resultBanner.className = `result-banner ${metrics.risk_level.toLowerCase()}`;
  riskTag.className = `risk-tag ${metrics.risk_level.toLowerCase()}`;
  riskTag.textContent = `${metrics.risk_level} RISK`;

  if (metrics.risk_level === 'LOW') {
    bannerSummary.textContent = 'Factual & Grounded — Response is consistent with expected knowledge.';
  } else if (metrics.risk_level === 'MEDIUM') {
    bannerSummary.textContent = 'Borderline Reliability — Minor ambiguities detected, acceptable without alert.';
  } else {
    bannerSummary.textContent = 'Hallucination Detected — Fabricated information requires verification.';
  }

  actionPill.className = `action-pill ${metrics.next_action.toLowerCase()}`;
  actionText.textContent = metrics.next_action;

  // Update Core Metrics
  valHalProb.textContent = metrics.hallucination_probability.toFixed(4);
  subHalProb.textContent = `${(metrics.hallucination_probability * 100).toFixed(2)}% likelihood`;
  valConfidence.textContent = metrics.confidence_score.toFixed(4);
  valLabel.textContent = metrics.predicted_label_name;
  valLatency.textContent = `${latencyMs} ms`;

  // Update Diagnostics
  diagModelSource.textContent = MODEL_ID;
  diagLoaded.textContent = 'true';
  diagExecuted.textContent = 'true';
  diagDegraded.textContent = 'false';
  diagLogits.textContent = `[${logits[0].toFixed(6)}, ${logits[1].toFixed(6)}]`;
  diagProbs.textContent = `[${probs[0].toFixed(6)}, ${probs[1].toFixed(6)}]`;
  diagInput.textContent = formattedInput;

  resultsCard.classList.add('visible');
  resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Collapsible Handlers
contextToggle.addEventListener('click', () => {
  const isHidden = contextWrapper.style.display === 'none';
  contextWrapper.style.display = isHidden ? 'block' : 'none';
  contextArrow.textContent = isHidden ? '▲ Hide' : '▼ Show';
});

diagToggle.addEventListener('click', () => {
  const isHidden = diagTable.style.display === 'none';
  diagTable.style.display = isHidden ? 'table' : 'none';
  diagArrow.textContent = isHidden ? '▲ Hide Details' : '▼ Show Details';
});

// Presets
presetTest1.addEventListener('click', () => {
  queryInput.value = 'What is the capital of France?';
  responseInput.value = 'The capital of France is Paris.';
  contextInput.value = '';
  contextWrapper.style.display = 'none';
  contextArrow.textContent = '▼ Show';
});

presetTest2.addEventListener('click', () => {
  queryInput.value = 'Who wrote Romeo and Juliet?';
  responseInput.value = 'Romeo and Juliet was written by Albert Einstein in 1920 in Germany.';
  contextInput.value = '';
  contextWrapper.style.display = 'none';
  contextArrow.textContent = '▼ Show';
});

presetClear.addEventListener('click', () => {
  queryInput.value = '';
  responseInput.value = '';
  contextInput.value = '';
  resultsCard.classList.remove('visible');
});

// Form listener
detectorForm.addEventListener('submit', handleDetection);

// Preload model on startup
getDetectorPipeline().catch(err => {
  console.warn('Initial model preload deferred until first run:', err);
});
