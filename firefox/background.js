/**
 * background.js — NoMusic background page (Firefox build)
 *
 * Firefox has neither chrome.tabCapture nor offscreen documents, so the pipeline is split differently
 * from the Chrome build:
 *   page <video>/<audio> -> media-hook.js (content script) -> port -> this page:
 *   FastEnhancer (main thread) -> stream-player worklet -> per-channel RNNoise -> analyser -> speakers
 *
 * Responsibilities:
 *  1. Host the audio engine (the background page is a real document in Firefox)
 *  2. Auto-follow the active tab and tell each tab's content script whether to filter or bypass
 *  3. Mute tabs the user leaves so only the active tab plays
 *  4. Answer the same popup messages as the Chrome service worker, so popup.js is shared
 */

'use strict';

const MODEL_URLS = {
  lite:     browser.runtime.getURL('assets/fastenhancer_t.wasm'),
  standard: browser.runtime.getURL('assets/fastenhancer_s.wasm')
};
const PLAYER_URL = browser.runtime.getURL('stream-player-worklet.js');
const RNNOISE_WASM_URL = browser.runtime.getURL('assets/rnnoise.wasm');
const RNNOISE_PROCESSOR_URL = browser.runtime.getURL('rnnoise-worklet.js');
const RNNOISE_ENABLED = true;
const SAMPLE_RATE = 48000;
const CHANNEL_COUNT = 2;
const HOP = 512;                  // samples per FastEnhancer hop (fe_hop_size() of the wasm modules)
const STARTUP_FADE_MS = 1000;
const OWNER_TIMEOUT_MS = 500;     // another frame may take over the engine after this much silence
const VIZ_WANTED_TTL_MS = 2500;
const MONO_RESYNC_HOPS = 100;     // see fastenhancer-worker.js

let extensionEnabled = true;
let continueAudioOnTabSwitch = false;
let activeTabId = null;
const mutedTabs = new Set();
const tabPorts = new Map();       // tabId -> Set<Port>

async function getModelUrl() {
  const r = await browser.storage.local.get(['nomusic_model']);
  return MODEL_URLS[r.nomusic_model === 'lite' ? 'lite' : 'standard'];
}

async function autoOnAllowed() {
  const { extensionEnabled: e = true, nomusic_autoOn: autoOn = true } =
    await browser.storage.local.get(['extensionEnabled', 'nomusic_autoOn']);
  extensionEnabled = e;
  return e && autoOn;
}

function isFilterable(url) {
  return url && (url.startsWith('http://') || url.startsWith('https://'));
}

// Only message the popup while it is open (it pings VIZ_WANTED from the moment it loads): Firefox reports
// a sendMessage with no receiver as an uncaught error even when the rejection is handled.
function popupOpen() {
  return Date.now() <= vizWantedUntil;
}

function sendStatus(state, extra) {
  if (!popupOpen()) return;
  browser.runtime.sendMessage({ type: 'DF_STATUS', state, ...(extra || {}) }).catch(() => {});
}

// ── Audio engine ─────────────────────────────────────────────────────────────

let audioCtx = null;
let playerNode = null;
let analyserNode = null;
let outputGainNode = null;
let engineBuilding = null;
let running = false;
let currentWet = 1;
let vizWantedUntil = 0;
let vizTimer = null;

let fe = null;                    // FastEnhancer wasm exports
let feModelUrl = null;
const channels = [];              // { handle, input, output } views into fe memory
let viewsBuffer = null;
const moduleCache = new Map();    // model URL -> Promise<WebAssembly.Module>

// Input staging: interleaved stereo, filled from content-script chunks one hop at a time.
const hopIn = new Float32Array(HOP * CHANNEL_COUNT);
let hopFill = 0;
let statesEqual = true;
let rightStale = false;
let monoRun = 0;
let ownerPort = null;
let ownerSeen = 0;

function loadModelModule(url) {
  let entry = moduleCache.get(url);
  if (!entry) {
    entry = fetch(url).then((r) => {
      if (!r.ok) throw new Error('HTTP ' + r.status + ' fetching ' + url);
      return r.arrayBuffer();
    }).then((bytes) => WebAssembly.compile(bytes));
    entry.catch(() => moduleCache.delete(url));
    moduleCache.set(url, entry);
  }
  return entry;
}

function refreshViews() {
  const buffer = fe.memory.buffer;
  if (buffer === viewsBuffer) return;
  viewsBuffer = buffer;
  for (const ch of channels) {
    ch.input = new Float32Array(buffer, fe.fe_stream_input(ch.handle), HOP);
    ch.output = new Float32Array(buffer, fe.fe_stream_output(ch.handle), HOP);
  }
}

function resetStreams() {
  hopFill = 0;
  statesEqual = true;
  rightStale = false;
  monoRun = 0;
  if (fe) for (const ch of channels) fe.fe_stream_reset(ch.handle);
}

async function loadModel(url) {
  if (fe && feModelUrl === url) return;
  const module = await loadModelModule(url);
  const instance = await WebAssembly.instantiate(module, {});
  const exports = instance.exports;
  if (exports.fe_hop_size() !== HOP) {
    throw new Error('FastEnhancer wasm hop size ' + exports.fe_hop_size() + ' != ' + HOP);
  }
  fe = exports;
  feModelUrl = url;
  viewsBuffer = null;
  channels.length = 0;
  for (let c = 0; c < CHANNEL_COUNT; c++) channels.push({ handle: fe.fe_stream_new(), input: null, output: null });
  refreshViews();
  // Warm-up hops so the first real frames don't pay for wasm tier-up.
  for (let i = 0; i < 3; i++) fe.fe_stream_process(channels[0].handle, 1);
  resetStreams();
  console.info('[NoMusic] FastEnhancer native WASM loaded:', url.split('/').pop());
}

async function createRnnoiseChain(ctx) {
  const response = await fetch(RNNOISE_WASM_URL);
  if (!response.ok) throw new Error('HTTP ' + response.status + ' fetching rnnoise.wasm');
  const wasmBytes = await response.arrayBuffer();
  const splitter = ctx.createChannelSplitter(CHANNEL_COUNT);
  const merger = ctx.createChannelMerger(CHANNEL_COUNT);
  const ready = [];

  for (let channel = 0; channel < CHANNEL_COUNT; channel++) {
    const node = new AudioWorkletNode(ctx, 'rnnoise-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
      channelCountMode: 'explicit',
      channelInterpretation: 'speakers',
      outputChannelCount: [1],
      processorOptions: { wasmBytes: wasmBytes.slice(0) }
    });
    ready.push(waitReady(node, 'RNNoise processor'));
    splitter.connect(node, channel, 0);
    node.connect(merger, 0, channel);
  }

  await Promise.all(ready);
  return { input: splitter, output: merger };
}

function waitReady(node, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(label + ' ready timeout')), 10000);
    node.port.onmessage = (event) => {
      if (event.data && event.data.type === 'PROCESSOR_READY') {
        clearTimeout(timer);
        node.port.onmessage = null;
        resolve();
      }
    };
    node.onprocessorerror = () => { clearTimeout(timer); reject(new Error(label + ' warm-up failed')); };
  });
}

async function buildGraph() {
  const ctx = new AudioContext({ sampleRate: SAMPLE_RATE, latencyHint: 'playback' });
  await Promise.all([
    ctx.audioWorklet.addModule(PLAYER_URL),
    RNNOISE_ENABLED ? ctx.audioWorklet.addModule(RNNOISE_PROCESSOR_URL) : null
  ]);

  const player = new AudioWorkletNode(ctx, 'stream-player-processor', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [CHANNEL_COUNT]
  });
  await waitReady(player, 'Stream player');
  const rnChain = RNNOISE_ENABLED ? await createRnnoiseChain(ctx) : null;

  const analyser = ctx.createAnalyser();
  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0.8;
  const gain = ctx.createGain();
  gain.gain.value = 0;

  if (rnChain) {
    player.connect(rnChain.input);
    rnChain.output.connect(analyser);
  } else {
    player.connect(analyser);
  }
  analyser.connect(gain);
  gain.connect(ctx.destination);

  audioCtx = ctx;
  playerNode = player;
  analyserNode = analyser;
  outputGainNode = gain;
}

async function startEngine(modelUrl) {
  sendStatus('loading', { step: 'model', progress: 0.2 });
  if (!audioCtx) {
    if (!engineBuilding) engineBuilding = buildGraph().finally(() => { engineBuilding = null; });
    await engineBuilding;
  }
  await loadModel(modelUrl);
  sendStatus('loading', { step: 'compiling', progress: 0.8 });

  resetStreams();
  ownerPort = null;
  playerNode.port.postMessage({ type: 'RESET' });
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  const now = audioCtx.currentTime;
  outputGainNode.gain.cancelScheduledValues(now);
  outputGainNode.gain.setValueAtTime(0, now);
  outputGainNode.gain.linearRampToValueAtTime(1, now + STARTUP_FADE_MS / 1000);
  running = true;
  startViz();
  sendStatus('active', { latency: 11, engine: feModelUrl.split('/').pop() });
}

async function stopEngine() {
  running = false;
  ownerPort = null;
  clearInterval(vizTimer);
  vizTimer = null;
  if (!audioCtx) return;
  outputGainNode.gain.cancelScheduledValues(audioCtx.currentTime);
  outputGainNode.gain.setValueAtTime(0, audioCtx.currentTime);
  playerNode.port.postMessage({ type: 'RESET' });
  resetStreams();
  // Keep the graph and model warm for the next start, but stop the worklets from burning CPU.
  await audioCtx.suspend().catch(() => {});
}

function startViz() {
  clearInterval(vizTimer);
  const freqData = new Uint8Array(analyserNode.frequencyBinCount);
  vizTimer = setInterval(() => {
    if (!running || !popupOpen()) return;
    analyserNode.getByteFrequencyData(freqData);
    browser.runtime.sendMessage({ type: 'DF_FREQ_DATA', data: Array.from(freqData) }).catch(() => {});
  }, 50);
}

// One hop of interleaved stereo in `hopIn` -> written interleaved into `out` at `offset`.
// Mono dedup as in fastenhancer-worker.js: dual-mono hops only run the left stream.
function processHop(out, offset) {
  refreshViews();
  const left = channels[0];
  const right = channels[1];
  let mono = true;
  for (let i = 0; i < HOP; i++) {
    const l = hopIn[i * 2];
    const r = hopIn[i * 2 + 1];
    left.input[i] = l;
    right.input[i] = r;
    if (l !== r) mono = false;
  }

  if (mono) {
    monoRun++;
    if (!statesEqual && monoRun >= MONO_RESYNC_HOPS) {
      statesEqual = true;
      rightStale = true;
    }
  } else {
    monoRun = 0;
  }

  if (mono && statesEqual) {
    fe.fe_stream_process(left.handle, currentWet);
    rightStale = true;
    for (let i = 0; i < HOP; i++) {
      out[offset + i * 2] = left.output[i];
      out[offset + i * 2 + 1] = left.output[i];
    }
    return;
  }

  if (rightStale) {
    fe.fe_stream_copy_state(right.handle, left.handle);
    rightStale = false;
  }
  statesEqual = false;
  fe.fe_stream_process(left.handle, currentWet);
  fe.fe_stream_process(right.handle, currentWet);
  for (let i = 0; i < HOP; i++) {
    out[offset + i * 2] = left.output[i];
    out[offset + i * 2 + 1] = right.output[i];
  }
}

function feed(port, samples) {
  if (!running || !fe || !(samples instanceof Float32Array)) return;

  // Only one frame of the tab drives the engine at a time; interleaving two streams would garble both.
  const now = Date.now();
  if (ownerPort !== port) {
    if (ownerPort && now - ownerSeen < OWNER_TIMEOUT_MS) return;
    ownerPort = port;
    hopFill = 0;
  }
  ownerSeen = now;

  const hops = Math.floor((hopFill + samples.length) / hopIn.length);
  const out = new Float32Array(hops * hopIn.length);
  let produced = 0;
  let pos = 0;
  while (pos < samples.length) {
    const n = Math.min(hopIn.length - hopFill, samples.length - pos);
    hopIn.set(samples.subarray(pos, pos + n), hopFill);
    hopFill += n;
    pos += n;
    if (hopFill === hopIn.length) {
      processHop(out, produced);
      produced += hopIn.length;
      hopFill = 0;
    }
  }
  if (produced) playerNode.port.postMessage({ type: 'PCM', samples: out }, [out.buffer]);
}

// ── Tab tracking ─────────────────────────────────────────────────────────────

function setTabMode(tabId, mode) {
  const ports = tabPorts.get(tabId);
  if (!ports) return;
  for (const port of ports) {
    try { port.postMessage({ type: 'MODE', mode }); } catch (_) {}
  }
}

function setMuted(tabId, muted) {
  if (muted) mutedTabs.add(tabId); else mutedTabs.delete(tabId);
  browser.tabs.update(tabId, { muted }).catch(() => {});
}

function unmuteAll() {
  for (const tabId of mutedTabs) browser.tabs.update(tabId, { muted: false }).catch(() => {});
  mutedTabs.clear();
}

function persistActiveTab() {
  browser.storage.session.set({ nomusic_activeTabId: activeTabId }).catch(() => {});
}

// Activations run one at a time; a newer request supersedes queued ones.
let serial = Promise.resolve();
function queue(fn) {
  serial = serial.then(fn).catch((err) => console.warn('[NoMusic]', err));
  return serial;
}

async function activateTab(tabId, { force = false, modelUrl } = {}) {
  if (tabId == null || (tabId === activeTabId && !force)) return;
  let tab;
  try { tab = await browser.tabs.get(tabId); } catch (_) { return; }
  if (!isFilterable(tab.url)) return;

  const oldId = activeTabId;
  if (oldId != null && oldId !== tabId) {
    setTabMode(oldId, 'bypass');
    setMuted(oldId, !continueAudioOnTabSwitch);
  }

  activeTabId = tabId;
  persistActiveTab();
  setMuted(tabId, false); // filtered audio plays from this page, not the tab
  try {
    await startEngine(modelUrl || await getModelUrl());
  } catch (err) {
    console.error('[NoMusic] engine start error:', err);
    activeTabId = null;
    persistActiveTab();
    await stopEngine();
    sendStatus('error', { error: err.message || String(err) });
    return;
  }
  if (activeTabId === tabId) setTabMode(tabId, 'filter');
}

async function deactivate() {
  if (activeTabId != null) {
    setTabMode(activeTabId, 'bypass');
    setMuted(activeTabId, false);
    activeTabId = null;
    persistActiveTab();
  }
  await stopEngine();
  sendStatus('stopped');
}

async function focusedTabId() {
  const [tab] = await browser.tabs.query({ active: true, lastFocusedWindow: true });
  return tab ? tab.id : null;
}

browser.tabs.onActivated.addListener(async ({ tabId }) => {
  if (continueAudioOnTabSwitch && activeTabId != null) return;
  if (!(await autoOnAllowed())) return;
  queue(() => activateTab(tabId));
});

browser.windows.onFocusChanged.addListener(async (windowId) => {
  if (windowId === browser.windows.WINDOW_ID_NONE) return;
  if (continueAudioOnTabSwitch && activeTabId != null) return;
  if (!(await autoOnAllowed())) return;
  const [tab] = await browser.tabs.query({ active: true, windowId }).catch(() => []);
  if (tab) queue(() => activateTab(tab.id));
});

browser.tabs.onRemoved.addListener((tabId) => {
  mutedTabs.delete(tabId);
  tabPorts.delete(tabId);
  if (tabId === activeTabId) queue(deactivate);
});

// ── Content-script ports ─────────────────────────────────────────────────────

browser.runtime.onConnect.addListener((port) => {
  if (port.name !== 'nomusic-media' || !port.sender || !port.sender.tab) return;
  const tabId = port.sender.tab.id;
  const isTopFrame = port.sender.frameId === 0;
  if (!tabPorts.has(tabId)) tabPorts.set(tabId, new Set());
  tabPorts.get(tabId).add(port);

  port.onMessage.addListener((msg) => {
    if (msg && msg.type === 'PCM' && tabId === activeTabId) feed(port, msg.samples);
  });

  port.onDisconnect.addListener(async () => {
    const ports = tabPorts.get(tabId);
    if (ports) { ports.delete(port); if (!ports.size) tabPorts.delete(tabId); }
    if (ownerPort === port) ownerPort = null;
    // The page was reloaded or left. With auto-on it is re-filtered when the new page connects;
    // without it (as in the Chrome build) filtering ends here.
    if (isTopFrame && tabId === activeTabId && !(await autoOnAllowed()) && tabId === activeTabId) {
      queue(deactivate);
    }
  });

  ready.then(async () => {
    if (tabId === activeTabId && running) {
      port.postMessage({ type: 'MODE', mode: 'filter' });
      return;
    }
    port.postMessage({ type: 'MODE', mode: 'bypass' });
    if (activeTabId == null && (await autoOnAllowed()) && tabId === (await focusedTabId())) {
      queue(() => activateTab(tabId));
    }
  });
});

// ── Popup messages (same protocol as the Chrome service worker) ─────────────

browser.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  switch (msg.type) {
    case 'VIZ_WANTED':
      vizWantedUntil = Date.now() + VIZ_WANTED_TTL_MS;
      return false;

    case 'GET_ACTIVE_TAB':
      browser.tabs.query({ active: true, currentWindow: true }).then((tabs) => {
        sendResponse(tabs && tabs[0]
          ? { tabId: tabs[0].id, title: tabs[0].title, url: tabs[0].url, favIconUrl: tabs[0].favIconUrl }
          : { error: 'No active tab' });
      });
      return true;

    case 'GET_ENGINE_STATE':
      ready.then(() => sendResponse({ activeTabId: running ? activeTabId : null, enabled: extensionEnabled }));
      return true;

    case 'TOGGLE_EXTENSION':
      extensionEnabled = typeof msg.enabled === 'boolean' ? msg.enabled : !extensionEnabled;
      browser.storage.local.set({ extensionEnabled });
      if (!extensionEnabled) {
        queue(deactivate);
        unmuteAll();
      }
      sendResponse({ ok: true, enabled: extensionEnabled });
      return false;

    case 'START_FILTERING':
      // No stream id on Firefox: the content script in that tab supplies the audio.
      queue(() => activateTab(msg.tabId, { force: true, modelUrl: msg.modelUrl }))
        .then(() => sendResponse({ ok: true }));
      return true;

    case 'STOP_FILTERING':
      queue(deactivate).then(() => sendResponse({ ok: true }));
      return true;

    case 'SET_LEVEL':
      currentWet = Math.max(0, Math.min(100, msg.value | 0)) / 100;
      sendResponse({ ok: true });
      return false;

    case 'SET_CONTINUE_AUDIO':
      continueAudioOnTabSwitch = !!msg.value;
      if (continueAudioOnTabSwitch) {
        unmuteAll();
      } else {
        for (const tabId of mutedTabs) browser.tabs.update(tabId, { muted: true }).catch(() => {});
      }
      sendResponse({ ok: true });
      return false;

    default:
      return false;
  }
});

// ── Startup ──────────────────────────────────────────────────────────────────

browser.runtime.onInstalled.addListener(() => {
  browser.storage.local.get(['extensionEnabled']).then((r) => {
    if (r.extensionEnabled === undefined) browser.storage.local.set({ extensionEnabled: true });
  });
});

// The event page can be unloaded when idle; restore the filtered tab when it starts again.
const ready = (async () => {
  const r = await browser.storage.local.get(['extensionEnabled', 'nomusic_continueAudio']);
  if (r.extensionEnabled !== undefined) extensionEnabled = r.extensionEnabled;
  if (r.nomusic_continueAudio !== undefined) continueAudioOnTabSwitch = r.nomusic_continueAudio;
  const s = await browser.storage.session.get(['nomusic_activeTabId']).catch(() => ({}));
  if (extensionEnabled && s.nomusic_activeTabId != null) {
    await queue(() => activateTab(s.nomusic_activeTabId, { force: true }));
  } else if (await autoOnAllowed()) {
    const tabId = await focusedTabId().catch(() => null);
    if (tabId != null) await queue(() => activateTab(tabId));
  }
})().catch(() => {});
