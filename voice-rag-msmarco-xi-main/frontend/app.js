/**
 * Dulset — Frontend Application
 * Features:
 *   - Real-time Speech-to-Text (Web Speech API)
 *   - Hybrid RAG Retrieval (Dense + BM25 + RRF)
 *   - Text-to-Speech Voice Output (ElevenLabs AI Voices + Browser Fallback)
 *   - Auto-speak Answer Option
 *   - Voice & ElevenLabs API Key Settings Management
 */

// ============================================================
// State
// ============================================================
const state = {
  language: 'en-US',       // default English
  isRecording: false,
  recognition: null,
  transcript: '',
  interimTranscript: '',
  isLoading: false,
  lastAnswerText: '',

  // Text-to-Speech State
  ttsProvider: localStorage.getItem('voicerag_tts_provider') || 'elevenlabs', // 'elevenlabs' | 'browser'
  elevenApiKey: localStorage.getItem('voicerag_eleven_api_key') || '',
  voiceId: localStorage.getItem('voicerag_voice_id') || 'EXAVITQu4vr4xnSDxMaL',
  modelId: localStorage.getItem('voicerag_model_id') || 'eleven_multilingual_v2',
  autoSpeak: localStorage.getItem('voicerag_auto_speak') === 'true',
  hasServerApiKey: false,
  isPlayingAudio: false,
  currentAudio: null,
  currentAudioUrl: null,
  availableVoices: [],
};

// ============================================================
// DOM Elements Cache
// ============================================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {};
function cacheElements() {
  els.micBtn = $('#mic-btn');
  els.micStatus = $('#mic-status');
  els.waveform = $('#waveform');
  els.transcriptArea = $('#transcript-area');
  els.textInput = $('#text-input');
  els.sendBtn = $('#send-btn');
  els.answerSection = $('#answer-section');
  els.answerStatus = $('#answer-status');
  els.answerText = $('#answer-text');
  els.chunksSection = $('#chunks-section');
  els.chunksList = $('#chunks-list');
  els.timingSection = $('#timing-section');
  els.timingGrid = $('#timing-grid');
  els.guardrailList = $('#guardrail-list');
  els.loadingOverlay = $('#loading-overlay');
  els.loadingText = $('#loading-text');
  els.toast = $('#toast');
  els.browserWarning = $('#browser-warning');
  els.totalLatency = $('#total-latency');

  // TTS & Settings Elements
  els.speakBtn = $('#speak-btn');
  els.stopBtn = $('#stop-btn');
  els.speakIcon = $('#speak-icon');
  els.speakBtnText = $('#speak-btn-text');
  els.ttsPlayingIndicator = $('#tts-playing-indicator');
  els.playingLabel = $('#playing-label');
  els.autoSpeakToggle = $('#auto-speak-toggle');
  els.voiceBadge = $('#voice-badge');
  els.openSettingsBtn = $('#open-settings-btn');

  // Voice Settings Modal Elements
  els.voiceModalBackdrop = $('#voice-modal-backdrop');
  els.closeModalBtn = $('#close-modal-btn');
  els.ttsProviderSelect = $('#tts-provider-select');
  els.elevenSettingsGroup = $('#eleven-settings-group');
  els.elevenApiKey = $('#eleven-api-key');
  els.toggleKeyVis = $('#toggle-key-vis');
  els.keyStatusHint = $('#key-status-hint');
  els.voiceSelectGroup = $('#voice-select-group');
  els.voiceSelect = $('#voice-select');
  els.modelSelectGroup = $('#model-select-group');
  els.modelSelect = $('#model-select');
  els.testVoiceBtn = $('#test-voice-btn');
  els.saveSettingsBtn = $('#save-settings-btn');
}

// ============================================================
// Web Speech API — Speech to Text (Input)
// ============================================================
function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    if (els.browserWarning) els.browserWarning.classList.add('visible');
    if (els.micBtn) {
      els.micBtn.style.opacity = '0.4';
      els.micBtn.style.cursor = 'not-allowed';
    }
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = state.language;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    state.isRecording = true;
    els.micBtn.classList.add('recording');
    els.micStatus.textContent = state.language === 'hi-IN' ? 'सुन रहा हूँ... बोलिए' : 'Listening... Speak now';
    els.micStatus.classList.add('recording-text');
    els.waveform.classList.add('active');
    animateWaveform();
  };

  recognition.onresult = (event) => {
    let interim = '';
    let final = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        final += transcript;
      } else {
        interim += transcript;
      }
    }
    if (final) {
      state.transcript += (state.transcript ? ' ' : '') + final;
    }
    state.interimTranscript = interim;
    updateTranscriptDisplay();
  };

  recognition.onerror = (event) => {
    if (event.error === 'no-speech') {
      return;
    }
    if (event.error === 'aborted') return;
    console.error('Speech recognition error:', event.error);
    const messages = {
      network: 'Chrome speech service is unavailable. Check your internet connection, then try again or type your question below.',
      'not-allowed': 'Microphone access is blocked. Allow microphone access for this site in the browser address bar, then try again.',
      'service-not-allowed': 'Chrome speech recognition is blocked. Allow microphone and speech access, then reload the page.',
      'audio-capture': 'No microphone was found. Connect a microphone or type your question below.',
    };
    showToast(messages[event.error] || `Speech recognition notice: ${event.error}. You can type your question below.`);
    stopRecording();
  };

  recognition.onend = () => {
    if (state.isRecording) {
      try {
        recognition.start();
      } catch (e) {
        stopRecording();
      }
      return;
    }
    stopRecording();
  };

  state.recognition = recognition;
}

function toggleRecording() {
  if (!state.recognition) {
    showToast('Speech recognition is not supported in this browser. Use Chrome or Edge.');
    return;
  }
  if (state.isRecording) {
    state.isRecording = false;
    state.recognition.stop();
    stopRecording();

    // Auto-send if we have a transcript
    if (state.transcript.trim()) {
      sendQuery(state.transcript.trim());
    }
  } else {
    stopSpeaking(); // Stop any active answer playback
    state.transcript = '';
    state.interimTranscript = '';
    updateTranscriptDisplay();
    hideResults();
    state.recognition.lang = state.language;
    try {
      state.recognition.start();
    } catch (e) {
      showToast('Could not start speech recognition. Check microphone permission or type your question below.');
    }
  }
}

function stopRecording() {
  state.isRecording = false;
  if (els.micBtn) els.micBtn.classList.remove('recording');
  if (els.micStatus) {
    els.micStatus.textContent = state.language === 'hi-IN'
      ? 'माइक्रोफ़ोन पर क्लिक करें या टाइप करें'
      : 'Click the microphone or type below';
    els.micStatus.classList.remove('recording-text');
  }
  if (els.waveform) els.waveform.classList.remove('active');
}

function updateTranscriptDisplay() {
  const area = els.transcriptArea;
  if (!area) return;
  if (!state.transcript && !state.interimTranscript) {
    area.innerHTML = '<span class="transcript-placeholder">' +
      (state.language === 'hi-IN' ? 'आपकी आवाज़ यहाँ दिखाई देगी...' : 'Your speech will appear here...') +
      '</span>';
    area.classList.remove('has-text');
    return;
  }
  area.classList.add('has-text');
  let html = '';
  if (state.transcript) {
    html += escapeHtml(state.transcript);
  }
  if (state.interimTranscript) {
    html += '<span class="transcript-interim"> ' + escapeHtml(state.interimTranscript) + '</span>';
  }
  area.innerHTML = html;
}

let waveInterval = null;
function animateWaveform() {
  if (waveInterval) clearInterval(waveInterval);
  const bars = els.waveform ? els.waveform.querySelectorAll('.wave-bar') : [];
  waveInterval = setInterval(() => {
    if (!state.isRecording) {
      clearInterval(waveInterval);
      return;
    }
    bars.forEach(bar => {
      bar.style.height = (6 + Math.random() * 28) + 'px';
    });
  }, 120);
}

// ============================================================
// Language Toggle
// ============================================================
function setLanguage(lang) {
  state.language = lang;
  $$('.lang-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = $(`[data-lang="${lang}"]`);
  if (activeBtn) activeBtn.classList.add('active');

  if (els.textInput) {
    els.textInput.placeholder = lang === 'hi-IN'
      ? 'यहाँ प्रश्न टाइप करें...'
      : 'Type your question here...';
  }
  updateTranscriptDisplay();
  if (!state.isRecording && els.micStatus) {
    els.micStatus.textContent = lang === 'hi-IN'
      ? 'माइक्रोफ़ोन पर क्लिक करें या टाइप करें'
      : 'Click the microphone or type below';
  }
}

// ============================================================
// Query API Call
// ============================================================
async function sendQuery(text) {
  if (!text.trim() || state.isLoading) return;

  stopSpeaking();
  state.isLoading = true;
  showLoading(state.language === 'hi-IN' ? 'उत्तर खोज रहा हूँ...' : 'Searching for answers...');
  hideResults();

  try {
    const response = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ text: text.trim(), language: state.language }),
    });

    let data;
    try {
      data = await response.json();
    } catch (pe) {
      data = null;
    }

    if (!response.ok) {
      if (data && (data.answer || data.error || data.detail)) {
        renderResult(data);
        return;
      }
      throw new Error(`Server responded with ${response.status}`);
    }

    if (!data) {
      throw new Error('Invalid response from server.');
    }

    renderResult(data);
  } catch (err) {
    console.error('Query error:', err);
    els.answerSection.classList.add('visible');
    els.answerStatus.textContent = '❌ Error';
    els.answerStatus.className = 'answer-status error';
    els.answerText.textContent = err.message || 'The server did not return an answer.';
    state.lastAnswerText = els.answerText.textContent;
    showToast(`Notice: ${err.message}.`);
  } finally {
    state.isLoading = false;
    hideLoading();
  }
}

function handleTextSubmit() {
  const text = els.textInput.value.trim();
  if (text) {
    state.transcript = text;
    updateTranscriptDisplay();
    sendQuery(text);
    els.textInput.value = '';
  }
}

// ============================================================
// Render Results
// ============================================================
function renderResult(data) {
  els.answerSection.classList.add('visible');

  const statusLabels = {
    answered: '✅ Answered',
    refused_unsafe: '🛡️ Refused — Unsafe',
    refused_off_topic: '🔍 Refused — Off Topic',
    refused_ungrounded: '⚠️ Refused — Ungrounded',
    error: '❌ Error',
  };
  els.answerStatus.textContent = statusLabels[data.status] || data.status;
  els.answerStatus.className = 'answer-status ' + data.status;

  let textToDisplay = '';
  if (data.answer) {
    textToDisplay = data.answer;
    state.lastAnswerText = data.answer;
    typeText(els.answerText, data.answer, () => {
      // If Auto-speak is enabled, speak after typing completes
      if (state.autoSpeak) {
        speakCurrentAnswer();
      }
    });
  } else {
    const refusalMessages = {
      refused_unsafe: "I can't help with that request.",
      refused_off_topic: "I can only answer questions grounded in the provided document collection.",
      refused_ungrounded: "I found related passages but couldn't produce a clearly supported answer.",
      error: "An error occurred while processing your query.",
    };
    textToDisplay = data.error || data.detail || refusalMessages[data.status] || 'No answer available.';
    state.lastAnswerText = textToDisplay;
    els.answerText.textContent = textToDisplay;

    if (state.autoSpeak) {
      speakCurrentAnswer();
    }
  }

  // Total latency
  if (data.total_latency_ms !== undefined) {
    els.totalLatency.textContent = `${data.total_latency_ms.toFixed(2)}ms`;
  }

  // Retrieved chunks
  if (data.retrieved && data.retrieved.length > 0) {
    els.chunksSection.classList.add('visible');
    els.chunksList.innerHTML = data.retrieved.map((r) => `
      <div class="chunk-item">
        <div class="chunk-meta">
          <span class="chunk-badge strategy">${escapeHtml(r.chunk.strategy)}</span>
          <span class="chunk-badge score">Fused: ${r.fused_score.toFixed(4)}</span>
          ${r.dense_score !== null ? `<span class="chunk-badge score">Dense: ${r.dense_score.toFixed(4)}</span>` : ''}
        </div>
        <div class="chunk-text">${escapeHtml(r.chunk.text)}</div>
      </div>
    `).join('');
  }

  // Timings
  if (data.timings && data.timings.length > 0) {
    els.timingSection.classList.add('visible');
    els.timingGrid.innerHTML = data.timings.map(t => `
      <div class="timing-item">
        <div class="timing-stage">${escapeHtml(t.stage)}</div>
        <div class="timing-value">${t.latency_ms.toFixed(2)}ms</div>
        <div class="${t.ok ? 'timing-ok' : 'timing-fail'}">
          ${t.ok ? '✓ OK' : '✗ Failed'}
          ${t.retries > 0 ? ` (${t.retries} retries)` : ''}
        </div>
      </div>
    `).join('');
  }

  // Guardrails
  if (data.guardrail_verdicts && data.guardrail_verdicts.length > 0) {
    els.guardrailList.innerHTML = data.guardrail_verdicts.map(v => `
      <div class="guardrail-badge ${v.passed ? 'passed' : 'failed'}">
        ${v.passed ? '✓' : '✗'} ${escapeHtml(v.stage)}
        ${v.reason ? ` — ${escapeHtml(v.reason)}` : ''}
      </div>
    `).join('');
  }
}

function typeText(element, text, onComplete) {
  element.innerHTML = '';
  let i = 0;
  const cursor = document.createElement('span');
  cursor.className = 'typing-cursor';

  function type() {
    if (i < text.length) {
      element.textContent = text.substring(0, i + 1);
      element.appendChild(cursor);
      i++;
      setTimeout(type, 10 + Math.random() * 14);
    } else {
      cursor.remove();
      if (typeof onComplete === 'function') {
        onComplete();
      }
    }
  }
  type();
}

function hideResults() {
  els.answerSection.classList.remove('visible');
  els.chunksSection.classList.remove('visible');
  els.timingSection.classList.remove('visible');
  stopSpeaking();
}

// ============================================================
// Text-to-Speech (TTS) & ElevenLabs Voice Engine
// ============================================================
async function initTTSVoices() {
  try {
    const res = await fetch('/api/tts/voices');
    if (res.ok) {
      const data = await res.json();
      state.hasServerApiKey = data.has_server_api_key;
      state.availableVoices = data.voices || [];
      if (data.default_voice_id && !localStorage.getItem('voicerag_voice_id')) {
        state.voiceId = data.default_voice_id;
      }
      populateVoiceDropdown(state.availableVoices);
      updateKeyHint();
      updateVoiceBadge();
    }
  } catch (err) {
    console.warn('Could not fetch TTS voices from backend:', err);
  }
}

function populateVoiceDropdown(voices) {
  if (!els.voiceSelect || !voices || voices.length === 0) return;
  els.voiceSelect.innerHTML = voices.map(v => `
    <option value="${v.id}" ${v.id === state.voiceId ? 'selected' : ''}>
      ${escapeHtml(v.name)} — ${escapeHtml(v.description)}
    </option>
  `).join('');
}

function updateVoiceBadge() {
  if (!els.voiceBadge) return;
  if (state.ttsProvider === 'browser') {
    els.voiceBadge.textContent = 'Browser Voice (Fallback)';
    return;
  }
  const currentVoice = state.availableVoices.find(v => v.id === state.voiceId);
  const voiceName = currentVoice ? currentVoice.name : 'Sarah';
  els.voiceBadge.textContent = `ElevenLabs (${voiceName})`;
}

function updateKeyHint() {
  if (!els.keyStatusHint) return;
  if (state.hasServerApiKey) {
    els.keyStatusHint.innerHTML = '✅ <strong>Server API Key active</strong> (set via <code>ELEVENLABS_API_KEY</code>). You can leave this blank or provide a custom key.';
  } else if (state.elevenApiKey) {
    els.keyStatusHint.innerHTML = '✅ <strong>Custom API Key configured</strong> in browser session.';
  } else {
    els.keyStatusHint.innerHTML = 'ℹ️ Enter your ElevenLabs API Key above, or set <code>ELEVENLABS_API_KEY</code> in <code>.env</code>.';
  }
}

async function speakCurrentAnswer() {
  const text = state.lastAnswerText || els.answerText.textContent.trim();
  if (!text) {
    showToast('No answer text to speak.');
    return;
  }

  // If already playing, stop
  if (state.isPlayingAudio) {
    stopSpeaking();
    return;
  }

  const effectiveKey = state.elevenApiKey || (state.hasServerApiKey ? 'server' : '');

  // If provider is ElevenLabs and no key is available, fallback or prompt user
  if (state.ttsProvider === 'elevenlabs' && !effectiveKey) {
    showToast('ElevenLabs API key not found. Using Browser Voice as fallback (or click Voice Settings to add your key).');
    speakWithBrowser(text);
    return;
  }

  if (state.ttsProvider === 'elevenlabs') {
    await speakWithElevenLabs(text);
  } else {
    speakWithBrowser(text);
  }
}

async function speakWithElevenLabs(text) {
  setPlayingState(true, 'elevenlabs');
  els.speakBtn.classList.add('loading');
  els.speakBtnText.textContent = 'Generating Voice...';

  try {
    const response = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        voice_id: state.voiceId,
        model_id: state.modelId,
        api_key: state.elevenApiKey || undefined,
      }),
    });

    if (!response.ok) {
      let errMsg = `Server error ${response.status}`;
      try {
        const errJson = await response.json();
        errMsg = errJson.detail || errJson.message || errMsg;
      } catch (e) {}
      throw new Error(errMsg);
    }

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    state.currentAudioUrl = audioUrl;

    const audio = new Audio(audioUrl);
    state.currentAudio = audio;

    audio.onended = () => {
      stopSpeaking();
    };

    audio.onerror = (e) => {
      console.error('Audio playback error:', e);
      showToast('Audio playback error. Falling back to browser speech.');
      stopSpeaking();
      speakWithBrowser(text);
    };

    await audio.play();
    els.speakBtn.classList.remove('loading');
    els.speakBtnText.textContent = 'Pause';
    els.speakIcon.textContent = '⏸️';
  } catch (err) {
    console.error('ElevenLabs TTS error:', err);
    showToast(`ElevenLabs notice: ${err.message}. Using Browser voice fallback.`);
    stopSpeaking();
    // Smooth fallback
    speakWithBrowser(text);
  }
}

function speakWithBrowser(text) {
  if (!('speechSynthesis' in window)) {
    showToast('Your browser does not support speech synthesis.');
    stopSpeaking();
    return;
  }

  window.speechSynthesis.cancel();
  setPlayingState(true, 'browser');
  els.speakBtnText.textContent = 'Stop Speaking';
  els.speakIcon.textContent = '⏹️';

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = state.language;
  utterance.rate = 1.0;
  utterance.pitch = 1.0;

  utterance.onend = () => {
    stopSpeaking();
  };

  utterance.onerror = (e) => {
    console.error('Browser speech synthesis error:', e);
    stopSpeaking();
  };

  state.currentAudio = utterance;
  window.speechSynthesis.speak(utterance);
}

function stopSpeaking() {
  if (state.currentAudio instanceof Audio) {
    try {
      state.currentAudio.pause();
      state.currentAudio.currentTime = 0;
    } catch (e) {}
  }
  if (state.currentAudioUrl) {
    try {
      URL.revokeObjectURL(state.currentAudioUrl);
    } catch (e) {}
    state.currentAudioUrl = null;
  }
  if ('speechSynthesis' in window) {
    try {
      window.speechSynthesis.cancel();
    } catch (e) {}
  }

  state.isPlayingAudio = false;
  state.currentAudio = null;
  setPlayingState(false);
}

function setPlayingState(isPlaying, provider = 'elevenlabs') {
  state.isPlayingAudio = isPlaying;
  if (!els.speakBtn) return;

  if (isPlaying) {
    els.speakBtn.classList.add('playing');
    if (els.stopBtn) els.stopBtn.style.display = 'inline-flex';
    if (els.ttsPlayingIndicator) {
      els.ttsPlayingIndicator.style.display = 'flex';
      const currentVoice = state.availableVoices.find(v => v.id === state.voiceId);
      const voiceName = currentVoice ? currentVoice.name : 'ElevenLabs';
      els.playingLabel.textContent = provider === 'elevenlabs'
        ? `Speaking with ElevenLabs (${voiceName})...`
        : 'Speaking with Browser Voice...';
    }
  } else {
    els.speakBtn.classList.remove('playing', 'loading');
    els.speakBtnText.textContent = 'Speak Answer';
    els.speakIcon.textContent = '🔊';
    if (els.stopBtn) els.stopBtn.style.display = 'none';
    if (els.ttsPlayingIndicator) els.ttsPlayingIndicator.style.display = 'none';
  }
}

// ============================================================
// Settings Modal & Preferences
// ============================================================
function openVoiceModal() {
  if (!els.voiceModalBackdrop) return;
  els.ttsProviderSelect.value = state.ttsProvider;
  els.elevenApiKey.value = state.elevenApiKey;
  els.voiceSelect.value = state.voiceId;
  els.modelSelect.value = state.modelId;
  toggleProviderFields();
  updateKeyHint();
  els.voiceModalBackdrop.classList.add('open');
}

function closeVoiceModal() {
  if (els.voiceModalBackdrop) {
    els.voiceModalBackdrop.classList.remove('open');
  }
}

function toggleProviderFields() {
  const isEleven = els.ttsProviderSelect.value === 'elevenlabs';
  if (els.elevenSettingsGroup) els.elevenSettingsGroup.style.display = isEleven ? 'flex' : 'none';
  if (els.voiceSelectGroup) els.voiceSelectGroup.style.display = isEleven ? 'flex' : 'none';
  if (els.modelSelectGroup) els.modelSelectGroup.style.display = isEleven ? 'flex' : 'none';
}

function saveVoiceSettings() {
  state.ttsProvider = els.ttsProviderSelect.value;
  state.elevenApiKey = els.elevenApiKey.value.trim();
  state.voiceId = els.voiceSelect.value;
  state.modelId = els.modelSelect.value;

  localStorage.setItem('voicerag_tts_provider', state.ttsProvider);
  localStorage.setItem('voicerag_eleven_api_key', state.elevenApiKey);
  localStorage.setItem('voicerag_voice_id', state.voiceId);
  localStorage.setItem('voicerag_model_id', state.modelId);

  updateVoiceBadge();
  updateKeyHint();
  closeVoiceModal();
  showToast('Voice settings saved successfully!');
}

async function testCurrentVoice() {
  const isEleven = els.ttsProviderSelect.value === 'elevenlabs';
  const key = els.elevenApiKey.value.trim() || (state.hasServerApiKey ? 'server' : '');
  const voiceId = els.voiceSelect.value;
  const modelId = els.modelSelect.value;
  const testText = "Hello! Dulset is ready with realistic speech synthesis.";

  if (isEleven && !key) {
    showToast('Please enter your ElevenLabs API Key first to test ElevenLabs voices.');
    return;
  }

  showToast('Playing voice sample...');
  if (isEleven) {
    try {
      const resp = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: testText,
          voice_id: voiceId,
          model_id: modelId,
          api_key: els.elevenApiKey.value.trim() || undefined,
        }),
      });
      if (!resp.ok) {
        let err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.play();
    } catch (e) {
      showToast(`Test failed: ${e.message}`);
    }
  } else {
    speakWithBrowser(testText);
  }
}

// ============================================================
// UI Helpers
// ============================================================
function showLoading(text) {
  if (els.loadingText) els.loadingText.textContent = text;
  if (els.loadingOverlay) els.loadingOverlay.classList.add('visible');
}

function hideLoading() {
  if (els.loadingOverlay) els.loadingOverlay.classList.remove('visible');
}

function showToast(message) {
  if (!els.toast) return;
  els.toast.textContent = message;
  els.toast.classList.add('visible');
  setTimeout(() => {
    els.toast.classList.remove('visible');
  }, 4500);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ============================================================
// Initialization & Event Listeners
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  cacheElements();
  initSpeechRecognition();
  initTTSVoices();

  // Mic button
  if (els.micBtn) els.micBtn.addEventListener('click', toggleRecording);

  // Send button
  if (els.sendBtn) els.sendBtn.addEventListener('click', handleTextSubmit);

  // Enter key in text input
  if (els.textInput) {
    els.textInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleTextSubmit();
      }
    });
  }

  // Language toggle buttons
  $$('.lang-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      setLanguage(btn.dataset.lang);
    });
  });

  // TTS Controls
  if (els.speakBtn) {
    els.speakBtn.addEventListener('click', speakCurrentAnswer);
  }
  if (els.stopBtn) {
    els.stopBtn.addEventListener('click', stopSpeaking);
  }
  if (els.autoSpeakToggle) {
    els.autoSpeakToggle.checked = state.autoSpeak;
    els.autoSpeakToggle.addEventListener('change', (e) => {
      state.autoSpeak = e.target.checked;
      localStorage.setItem('voicerag_auto_speak', state.autoSpeak);
      showToast(state.autoSpeak ? 'Auto-speak enabled for answers' : 'Auto-speak disabled');
    });
  }
  if (els.voiceBadge) {
    els.voiceBadge.addEventListener('click', openVoiceModal);
  }

  // Voice Settings Modal
  if (els.openSettingsBtn) {
    els.openSettingsBtn.addEventListener('click', openVoiceModal);
  }
  if (els.closeModalBtn) {
    els.closeModalBtn.addEventListener('click', closeVoiceModal);
  }
  if (els.voiceModalBackdrop) {
    els.voiceModalBackdrop.addEventListener('click', (e) => {
      if (e.target === els.voiceModalBackdrop) closeVoiceModal();
    });
  }
  if (els.ttsProviderSelect) {
    els.ttsProviderSelect.addEventListener('change', toggleProviderFields);
  }
  if (els.toggleKeyVis) {
    els.toggleKeyVis.addEventListener('click', () => {
      if (els.elevenApiKey.type === 'password') {
        els.elevenApiKey.type = 'text';
        els.toggleKeyVis.textContent = '🔒';
      } else {
        els.elevenApiKey.type = 'password';
        els.toggleKeyVis.textContent = '👁️';
      }
    });
  }
  if (els.saveSettingsBtn) {
    els.saveSettingsBtn.addEventListener('click', saveVoiceSettings);
  }
  if (els.testVoiceBtn) {
    els.testVoiceBtn.addEventListener('click', testCurrentVoice);
  }

  // Set initial language and state
  setLanguage('en-US');
  updateTranscriptDisplay();
  updateVoiceBadge();
});
