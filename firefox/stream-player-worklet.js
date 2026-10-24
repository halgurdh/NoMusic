/**
 * stream-player-worklet.js — plays processed audio pushed in over the node's port (Firefox build).
 *
 * Firefox has no SharedArrayBuffer in extension pages, so the background posts interleaved stereo
 * Float32Array chunks here instead of sharing a ring buffer. Playback starts once PREBUFFER_FRAMES are
 * queued (absorbing the bursty arrival from the page), re-prebuffers after an underrun, and drops the
 * oldest audio when the queue grows past MAX_FRAMES so latency stays bounded.
 */

'use strict';

const CAPACITY_FRAMES = 48000;   // 1 s ring
const PREBUFFER_FRAMES = 4096;   // ~85 ms
const MAX_FRAMES = 14400;        // ~300 ms

class StreamPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ring = new Float32Array(CAPACITY_FRAMES * 2);
    this.read = 0;    // frame index
    this.write = 0;   // frame index
    this.count = 0;   // frames queued
    this.playing = false;

    this.port.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === 'PCM') this._push(data.samples);
      else if (data.type === 'RESET') { this.read = this.write = this.count = 0; this.playing = false; }
    };
    this.port.postMessage({ type: 'PROCESSOR_READY' });
  }

  _push(samples) {
    const frames = samples.length >> 1;
    for (let i = 0; i < frames; i++) {
      const base = this.write * 2;
      this.ring[base] = samples[i * 2];
      this.ring[base + 1] = samples[i * 2 + 1];
      this.write = (this.write + 1) % CAPACITY_FRAMES;
    }
    this.count += frames;
    if (this.count > MAX_FRAMES) {
      const drop = this.count - PREBUFFER_FRAMES;
      this.read = (this.read + drop) % CAPACITY_FRAMES;
      this.count -= drop;
    }
  }

  process(_inputs, outputs) {
    const left = outputs[0][0];
    const right = outputs[0][1] || left;
    const n = left.length;

    if (!this.playing && this.count >= PREBUFFER_FRAMES) this.playing = true;
    if (!this.playing || this.count < n) {
      if (this.playing) this.playing = false; // underrun: wait for a fresh prebuffer
      left.fill(0);
      if (right !== left) right.fill(0);
      return true;
    }

    for (let i = 0; i < n; i++) {
      const base = this.read * 2;
      left[i] = this.ring[base];
      right[i] = this.ring[base + 1];
      this.read = (this.read + 1) % CAPACITY_FRAMES;
    }
    this.count -= n;
    return true;
  }
}

registerProcessor('stream-player-processor', StreamPlayerProcessor);
