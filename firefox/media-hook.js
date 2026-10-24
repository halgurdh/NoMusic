/**
 * media-hook.js — NoMusic content script (Firefox build)
 *
 * Firefox has no tabCapture, so instead of capturing the whole tab we reroute the page's own <video> and
 * <audio> elements through Web Audio. While this tab is the filtered one:
 *   media elements -> capture node -> (port) -> background engine -> speakers
 * and the elements produce no sound of their own. In "bypass" mode the hooked elements play straight to
 * the speakers again (createMediaElementSource can't be undone, so we only hook once filtering is wanted).
 */

(function () {
  'use strict';

  if (globalThis.__noMusicMediaHook) return;
  globalThis.__noMusicMediaHook = true;

  const SAMPLE_RATE = 48000;
  const CHUNK_FRAMES = 2048;      // ScriptProcessor buffer: ~43 ms per message
  const PING_MS = 20000;          // keeps the background event page awake while this tab is filtered

  let port = null;
  let mode = 'bypass';            // 'filter' | 'bypass' — told by the background
  let ctx = null;
  let captureNode = null;
  const hooked = new Map();       // element -> MediaElementAudioSourceNode
  let pingTimer = null;

  // ── Background connection ─────────────────────────────────────────────────

  function connect() {
    if (port) return port;
    try {
      port = browser.runtime.connect({ name: 'nomusic-media' });
    } catch (_) {
      port = null; // extension reloaded / context gone
      return null;
    }
    port.onMessage.addListener(onMessage);
    port.onDisconnect.addListener(() => {
      port = null;
      // The background event page went away. If we were filtering, reconnect so it wakes up and
      // re-sends our mode; until then keep the elements silent rather than leaking music.
      if (mode === 'filter') setTimeout(connect, 250);
    });
    return port;
  }

  function send(msg) {
    const p = connect();
    if (!p) return;
    try { p.postMessage(msg); } catch (_) { port = null; }
  }

  function onMessage(msg) {
    if (!msg || msg.type !== 'MODE') return;
    setMode(msg.mode);
  }

  // ── Mode switching ────────────────────────────────────────────────────────

  function setMode(next) {
    if (next === mode) return;
    mode = next;
    if (mode === 'filter') {
      for (const el of document.querySelectorAll('video, audio')) hook(el);
      clearInterval(pingTimer);
      pingTimer = setInterval(() => send({ type: 'PING' }), PING_MS);
    } else {
      clearInterval(pingTimer);
      pingTimer = null;
    }
    for (const source of hooked.values()) route(source);
  }

  function route(source) {
    try { source.disconnect(); } catch (_) {}
    source.connect(mode === 'filter' ? captureNode : ctx.destination);
  }

  // ── Audio graph ───────────────────────────────────────────────────────────

  function ensureContext() {
    if (ctx) return ctx;
    ctx = new AudioContext({ sampleRate: SAMPLE_RATE, latencyHint: 'playback' });
    captureNode = ctx.createScriptProcessor(CHUNK_FRAMES, 2, 2);
    captureNode.channelCountMode = 'explicit';
    captureNode.channelInterpretation = 'speakers';

    const left = new Float32Array(CHUNK_FRAMES);
    const right = new Float32Array(CHUNK_FRAMES);
    captureNode.onaudioprocess = (event) => {
      // The output buffer is left silent: the page itself must not play the unfiltered audio.
      if (mode !== 'filter' || !anyPlaying()) return;
      event.inputBuffer.copyFromChannel(left, 0);
      event.inputBuffer.copyFromChannel(right, 1);
      const samples = new Float32Array(CHUNK_FRAMES * 2);
      for (let i = 0; i < CHUNK_FRAMES; i++) {
        samples[i * 2] = left[i];
        samples[i * 2 + 1] = right[i];
      }
      send({ type: 'PCM', samples });
    };
    // A ScriptProcessor only runs while connected to the destination.
    captureNode.connect(ctx.destination);
    return ctx;
  }

  function anyPlaying() {
    for (const el of hooked.keys()) {
      if (!el.paused && !el.ended && !el.muted && el.volume > 0) return true;
    }
    return false;
  }

  // createMediaElementSource outputs silence for cross-origin media loaded without CORS, which would mute
  // the video entirely — leave those elements alone (they play unfiltered).
  function canHook(el) {
    if (el.srcObject) return false;
    const src = el.currentSrc || el.src;
    if (!src) return true; // no source yet (e.g. MSE attaches later)
    if (src.startsWith('blob:') || src.startsWith('data:')) return true;
    if (el.crossOrigin != null) return true;
    try { return new URL(src, location.href).origin === location.origin; } catch (_) { return false; }
  }

  function hook(el) {
    if (mode !== 'filter' || hooked.has(el) || !canHook(el)) return;
    try {
      ensureContext();
      const source = ctx.createMediaElementSource(el);
      hooked.set(el, source);
      route(source);
    } catch (err) {
      // Already captured by the page's own AudioContext, or similar — play it unfiltered.
      console.debug('[NoMusic] could not hook media element:', err && err.message);
    }
    if (ctx.state === 'suspended') ctx.resume().catch(() => {});
  }

  // `play` doesn't bubble, but capture-phase listeners on the document see it for every element in the
  // document (including ones created later). Elements never inserted into the DOM aren't covered.
  document.addEventListener('play', (event) => {
    const el = event.target;
    if (!(el instanceof HTMLMediaElement)) return;
    connect(); // tells the background this tab is playing media
    hook(el);
    if (ctx && ctx.state === 'suspended') ctx.resume().catch(() => {});
  }, true);

  // Announce ourselves so the background can tell us whether this tab is the filtered one.
  connect();
})();
