/**
 * fe-worklet.js — FastEnhancer as an AudioWorkletProcessor (Firefox build).
 *
 * Runs inside the page's own AudioContext, on the real-time audio thread of the tab that is playing.
 * (The background page's process is throttled by Windows efficiency mode, which made processing there
 * stutter.) The models' wasm bytes arrive via processorOptions and are compiled here.
 *
 * Firefox runs worklet wasm in its baseline tier until a function has been called a few hundred times,
 * and only modules compiled asynchronously are tiered up at all. Baseline Standard costs ~15-25 ms per
 * 10.7 ms hop (slow motion), optimized ~3 ms; Lite is cheap in either tier. So when Standard is selected:
 *   1. Lite filters from the start;
 *   2. Standard is compiled asynchronously and warmed up on every WARM_EVERY-th hop (off the output path);
 *   3. once Standard hops average under FAST_MS (typically after 30-60 s), Lite crossfades into Standard.
 *
 * 128-frame render quanta are collected into 512-sample hops; output lags input by one hop.
 */

'use strict';

const HOP = 512;
const QUANTUM = 128;
const MONO_RESYNC_HOPS = 100;
const WARM_EVERY = 2;        // warm-up hop rate while Lite is filtering (one channel, so ~half a hop's cost)
const PROBE_HOPS = 32;       // Standard hops averaged per speed check (the worklet clock is coarse)
const FAST_MS = 5;           // under half the hop budget: safe to switch
const CROSSFADE_HOPS = 48;   // ~0.5 s

// One FastEnhancer module with a stream per channel, plus the mono dedup from fastenhancer-worker.js:
// dual-mono hops only run the left stream.
class Enhancer {
  constructor(instance) {
    const fe = instance.exports;
    if (fe.fe_hop_size() !== HOP) throw new Error('FastEnhancer hop size ' + fe.fe_hop_size());
    this.fe = fe;
    this.left = fe.fe_stream_new();
    this.right = fe.fe_stream_new();
    this.viewsBuffer = null;
    this.outL = new Float32Array(HOP);
    this.outR = new Float32Array(HOP);
    this.reset();
  }

  reset() {
    this.fe.fe_stream_reset(this.left);
    this.fe.fe_stream_reset(this.right);
    this.statesEqual = true;
    this.rightStale = false;
    this.monoRun = 0;
  }

  _views() {
    const buffer = this.fe.memory.buffer;
    if (buffer !== this.viewsBuffer) {
      this.viewsBuffer = buffer;
      this.lIn = new Float32Array(buffer, this.fe.fe_stream_input(this.left), HOP);
      this.lOut = new Float32Array(buffer, this.fe.fe_stream_output(this.left), HOP);
      this.rIn = new Float32Array(buffer, this.fe.fe_stream_input(this.right), HOP);
      this.rOut = new Float32Array(buffer, this.fe.fe_stream_output(this.right), HOP);
    }
  }

  // Processes one hop into this.outL / this.outR.
  run(inL, inR, wet) {
    const fe = this.fe;
    this._views();
    let mono = true;
    for (let i = 0; i < HOP; i++) {
      if (inL[i] !== inR[i]) { mono = false; break; }
    }
    this.lIn.set(inL);
    this.rIn.set(inR);

    if (mono) {
      this.monoRun++;
      if (!this.statesEqual && this.monoRun >= MONO_RESYNC_HOPS) {
        this.statesEqual = true;
        this.rightStale = true;
      }
    } else {
      this.monoRun = 0;
    }

    if (mono && this.statesEqual) {
      fe.fe_stream_process(this.left, wet);
      this.rightStale = true;
      this.outL.set(this.lOut);
      this.outR.set(this.lOut);
      return;
    }

    if (this.rightStale) {
      fe.fe_stream_copy_state(this.right, this.left);
      this.rightStale = false;
    }
    this.statesEqual = false;
    fe.fe_stream_process(this.left, wet);
    fe.fe_stream_process(this.right, wet);
    this.outL.set(this.lOut);
    this.outR.set(this.rOut);
  }

  // One throwaway hop on the left stream (warm-up only); returns its duration in ms.
  probe(inL) {
    this._views();
    this.lIn.set(inL);
    const t0 = Date.now();
    this.fe.fe_stream_process(this.left, 1);
    return Date.now() - t0;
  }
}

class FastEnhancerProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { bytes, liteBytes, wet } = options.processorOptions || {};
    this.wet = typeof wet === 'number' ? wet : 1;
    this.inL = new Float32Array(HOP);
    this.inR = new Float32Array(HOP);
    this.outL = new Float32Array(HOP);
    this.outR = new Float32Array(HOP);
    this.fill = 0;
    this.active = null;      // Enhancer whose output is played
    this.pending = null;     // Standard, while warming up
    this.hopCount = 0;
    this.probeMs = 0;
    this.probeCount = 0;
    this.fadeHop = -1;       // >= 0 while crossfading active -> pending

    this.port.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === 'SET_WET') this.wet = Math.max(0, Math.min(1, data.wet));
      else if (data.type === 'RESET') this._reset();
    };

    const fail = (err) => this.port.postMessage({ type: 'ERROR', error: err.message || String(err) });
    try {
      if (liteBytes) {
        // Lite is cheap even unoptimized, so it can start right away (synchronous compile).
        this.active = new Enhancer(new WebAssembly.Instance(new WebAssembly.Module(liteBytes), {}));
        this.port.postMessage({ type: 'PROCESSOR_READY', model: 'lite' });
        WebAssembly.instantiate(bytes, {}).then(({ instance }) => {
          this.pending = new Enhancer(instance);
        }).catch(fail);
      } else {
        WebAssembly.instantiate(bytes, {}).then(({ instance }) => {
          this.active = new Enhancer(instance);
          this.port.postMessage({ type: 'PROCESSOR_READY' });
        }).catch(fail);
      }
    } catch (err) {
      fail(err);
    }
  }

  _reset() {
    this.fill = 0;
    this.outL.fill(0);
    this.outR.fill(0);
    if (this.active) this.active.reset();
  }

  _runHop() {
    const active = this.active;
    const pending = this.pending;
    active.run(this.inL, this.inR, this.wet);

    if (pending && this.fadeHop < 0) {
      if (++this.hopCount % WARM_EVERY === 0) {
        this.probeMs += pending.probe(this.inL);
        if (++this.probeCount === PROBE_HOPS) {
          if (this.probeMs / PROBE_HOPS < FAST_MS) {
            pending.reset();
            this.fadeHop = 0;
          }
          this.probeMs = 0;
          this.probeCount = 0;
        }
      }
    }

    if (this.fadeHop < 0) {
      this.outL.set(active.outL);
      this.outR.set(active.outR);
      return;
    }

    // Crossfade Lite -> Standard; Standard's recurrent state settles during the fade.
    pending.run(this.inL, this.inR, this.wet);
    for (let i = 0; i < HOP; i++) {
      const g = Math.min(1, (this.fadeHop * HOP + i) / (CROSSFADE_HOPS * HOP));
      this.outL[i] = active.outL[i] * (1 - g) + pending.outL[i] * g;
      this.outR[i] = active.outR[i] * (1 - g) + pending.outR[i] * g;
    }
    if (++this.fadeHop >= CROSSFADE_HOPS) {
      this.active = pending;
      this.pending = null;
      this.fadeHop = -1;
      this.port.postMessage({ type: 'MODEL_SWITCHED', model: 'standard' });
    }
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const outL = outputs[0][0];
    const outR = outputs[0][1] || outL;
    // No input connected (bypass mode) or not loaded yet: nothing to do, stay alive.
    if (!this.active || !input || !input[0]) {
      outL.fill(0);
      if (outR !== outL) outR.fill(0);
      return true;
    }

    const inL = input[0];
    const inR = input[1] || inL;
    // Emit the previous hop's output for this slot, then overwrite the slot with new input.
    outL.set(this.outL.subarray(this.fill, this.fill + QUANTUM));
    if (outR !== outL) outR.set(this.outR.subarray(this.fill, this.fill + QUANTUM));
    this.inL.set(inL, this.fill);
    this.inR.set(inR, this.fill);
    this.fill += QUANTUM;
    if (this.fill === HOP) {
      this._runHop();
      this.fill = 0;
    }
    return true;
  }
}

registerProcessor('fastenhancer-processor', FastEnhancerProcessor);
