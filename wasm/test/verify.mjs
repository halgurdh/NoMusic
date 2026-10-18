// Verifies the standalone FastEnhancer WASM modules against onnxruntime reference output
// (produced by make_reference.py) and benchmarks per-hop latency.
//   node wasm/test/verify.mjs [path/to/assets]
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const assets = process.argv[2] || join(here, '..', '..', 'assets');
let failed = false;

function bench(e, s) {
  for (let i = 0; i < 100; i++) e.fe_stream_process(s, 1.0);
  const runs = [];
  for (let r = 0; r < 7; r++) {
    const t0 = performance.now();
    for (let i = 0; i < 150; i++) e.fe_stream_process(s, 1.0);
    runs.push((performance.now() - t0) / 150);
  }
  return runs.sort((a, b) => a - b)[3];
}

for (const tag of ['t', 's']) {
  const meta = JSON.parse(readFileSync(join(here, 'ref', `${tag}.json`), 'utf8'));
  const x = new Float32Array(readFileSync(join(here, 'ref', `${tag}_in.f32`)).buffer.slice(0));
  const ref = new Float32Array(readFileSync(join(here, 'ref', `${tag}_out.f32`)).buffer.slice(0));

  const file = join(assets, `fastenhancer_${tag}.wasm`);
  if (!existsSync(file)) continue;
  const bytes = readFileSync(file);
  if (!WebAssembly.validate(bytes)) {
    console.log(`SKIP ${file}: not supported by this engine`);
    continue;
  }
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const e = instance.exports;
  const hop = e.fe_hop_size();
  if (hop !== meta.hop) throw new Error(`hop mismatch ${hop} vs ${meta.hop}`);

  const s = e.fe_stream_new();
  const mem = () => e.memory.buffer;
  const y = new Float32Array(x.length);
  for (let h = 0; h < meta.hops; h++) {
    new Float32Array(mem(), e.fe_stream_input(s), hop).set(x.subarray(h * hop, (h + 1) * hop));
    e.fe_stream_process(s, 1.0);
    y.set(new Float32Array(mem(), e.fe_stream_output(s), hop), h * hop);
  }

  let maxErr = 0, errE = 0, sigE = 0;
  for (let i = 0; i < y.length; i++) {
    const d = y[i] - ref[i];
    maxErr = Math.max(maxErr, Math.abs(d));
    errE += d * d;
    sigE += ref[i] * ref[i];
  }
  const snr = 10 * Math.log10(sigE / Math.max(errE, 1e-30));

  let cacheErr = 0;
  for (let k = 0; k < meta.caches.length; k++) {
    const c = new Float32Array(mem(), e.fe_stream_cache(s, k), e.fe_cache_len(k));
    meta.final_caches[k].forEach((v, i) => { cacheErr = Math.max(cacheErr, Math.abs(v - c[i])); });
  }

  // copy_state must reproduce the source stream exactly
  const s2 = e.fe_stream_new();
  e.fe_stream_copy_state(s2, s);
  const probe = x.subarray(0, hop);
  new Float32Array(mem(), e.fe_stream_input(s), hop).set(probe);
  new Float32Array(mem(), e.fe_stream_input(s2), hop).set(probe);
  e.fe_stream_process(s, 1.0);
  e.fe_stream_process(s2, 1.0);
  const o1 = new Float32Array(mem(), e.fe_stream_output(s), hop);
  const o2 = new Float32Array(mem(), e.fe_stream_output(s2), hop);
  const copyOk = o1.every((v, i) => v === o2[i]);

  const ms = bench(e, s);
  const budget = (hop / 48000) * 1000 / 2; // two channels share one hop period

  const ok = snr > 60 && maxErr < 1e-3 && cacheErr < 1e-3 && copyOk;
  failed ||= !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'} fastenhancer_${tag}.wasm (${bytes.length} B): ` +
    `SNR vs ORT ${snr.toFixed(1)} dB, max |err| ${maxErr.toExponential(2)}, cache max |err| ${cacheErr.toExponential(2)}, ` +
    `copy_state ${copyOk ? 'ok' : 'MISMATCH'}; ${ms.toFixed(3)} ms/hop (${(budget / ms).toFixed(1)}x real-time for stereo)`);
  e.fe_stream_free(s);
  e.fe_stream_free(s2);
}

process.exit(failed ? 1 : 0);
