// Per-kernel profile of a wasm built with `--features profile`.
//   node wasm/test/profile.mjs path/to/fastenhancer_X.wasm [hops]
import { readFileSync } from 'node:fs';

const file = process.argv[2];
const hops = Number(process.argv[3] || 300);
const { instance } = await WebAssembly.instantiate(readFileSync(file), { env: { fe_now: () => performance.now() } });
const e = instance.exports;
const s = e.fe_stream_new();
const input = new Float32Array(e.memory.buffer, e.fe_stream_input(s), e.fe_hop_size());
for (let i = 0; i < 100; i++) { input.forEach((_, j) => { input[j] = Math.sin(i * 512 + j) * 0.3; }); e.fe_stream_process(s, 1); }

const n = e.fe_prof_len();
new Float64Array(e.memory.buffer, e.fe_prof_times(), n).fill(0);
const t0 = performance.now();
for (let i = 0; i < hops; i++) e.fe_stream_process(s, 1);
const total = (performance.now() - t0) / hops;

const times = new Float64Array(e.memory.buffer, e.fe_prof_times(), n);
const dec = new TextDecoder();
const byKernel = new Map(), byStep = [];
for (let i = 0; i < n; i++) {
  const label = dec.decode(new Uint8Array(e.memory.buffer, e.fe_prof_label_ptr(i), e.fe_prof_label_len(i)));
  const ms = times[i] / hops;
  byStep.push([ms, i, label]);
  const k = label.split(':')[1];
  byKernel.set(k, (byKernel.get(k) || 0) + ms);
}
console.log(`total ${total.toFixed(3)} ms/hop (instrumented)`);
console.log('-- by kernel');
[...byKernel].sort((a, b) => b[1] - a[1]).forEach(([k, ms]) => console.log(`${(100 * ms / total).toFixed(1).padStart(5)}%  ${ms.toFixed(3)} ms  ${k}`));
console.log('-- top steps');
byStep.sort((a, b) => b[0] - a[0]).slice(0, 15).forEach(([ms, i, l]) => console.log(`${(100 * ms / total).toFixed(1).padStart(5)}%  #${i} ${l}`));
