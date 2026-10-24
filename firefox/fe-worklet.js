/**
 * fe-worklet.js — FastEnhancer as an AudioWorkletProcessor (Firefox build).
 *
 * Runs inside the page's own AudioContext, on the real-time audio thread of the tab that is playing.
 * (The background page's process is throttled by Windows efficiency mode, which made processing there
 * stutter.) The model's wasm bytes arrive via processorOptions and are compiled here.
 *
 * 128-frame render quanta are collected into 512-sample hops; output lags input by one hop.
 * Mono dedup as in fastenhancer-worker.js: dual-mono hops only run the left stream.
 */

'use strict';

const HOP = 512;
const QUANTUM = 128;
const MONO_RESYNC_HOPS = 100;

class FastEnhancerProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { bytes, wet } = options.processorOptions || {};
    this.wet = typeof wet === 'number' ? wet : 1;
    this.inL = new Float32Array(HOP);
    this.inR = new Float32Array(HOP);
    this.outL = new Float32Array(HOP);
    this.outR = new Float32Array(HOP);
    this.fill = 0;
    this.statesEqual = true;
    this.rightStale = false;
    this.monoRun = 0;
    this.fe = null;

    this.port.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === 'SET_WET') this.wet = Math.max(0, Math.min(1, data.wet));
      else if (data.type === 'RESET') this._reset();
    };

    try {
      const fe = new WebAssembly.Instance(new WebAssembly.Module(bytes), {}).exports;
      if (fe.fe_hop_size() !== HOP) throw new Error('FastEnhancer hop size ' + fe.fe_hop_size());
      this.fe = fe;
      this.left = fe.fe_stream_new();
      this.right = fe.fe_stream_new();
      this.viewsBuffer = null;
      // Warm-up hops so the first real audio doesn't pay for wasm tier-up.
      for (let i = 0; i < 3; i++) fe.fe_stream_process(this.left, 1);
      this._reset();
      this.port.postMessage({ type: 'PROCESSOR_READY' });
    } catch (err) {
      this.port.postMessage({ type: 'ERROR', error: err.message || String(err) });
    }
  }

  _reset() {
    this.fill = 0;
    this.outL.fill(0);
    this.outR.fill(0);
    this.statesEqual = true;
    this.rightStale = false;
    this.monoRun = 0;
    if (this.fe) {
      this.fe.fe_stream_reset(this.left);
      this.fe.fe_stream_reset(this.right);
    }
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

  _runHop() {
    const fe = this.fe;
    this._views();
    let mono = true;
    for (let i = 0; i < HOP; i++) {
      if (this.inL[i] !== this.inR[i]) { mono = false; break; }
    }
    this.lIn.set(this.inL);
    this.rIn.set(this.inR);

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
      fe.fe_stream_process(this.left, this.wet);
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
    fe.fe_stream_process(this.left, this.wet);
    fe.fe_stream_process(this.right, this.wet);
    this.outL.set(this.lOut);
    this.outR.set(this.rOut);
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const outL = outputs[0][0];
    const outR = outputs[0][1] || outL;
    // No input connected (bypass mode): nothing to do, stay alive.
    if (!this.fe || !input || !input[0]) {
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
