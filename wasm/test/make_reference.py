#!/usr/bin/env python3
"""Run the original ONNX models with onnxruntime in streaming mode and dump reference I/O
for verify.mjs.  Output: test/ref/<tag>.json + <tag>_in.f32 + <tag>_out.f32"""

import json
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, '..', 'models')
OUT = os.path.join(HERE, 'ref')
HOPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SR = 48000


def test_signal(n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    # "speech": vibrato harmonic source with syllable-rate amplitude modulation
    f0 = 140 + 25 * np.sin(2 * np.pi * 3.1 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    voice = sum(np.sin(k * ph) / k for k in range(1, 18))
    voice *= np.clip(np.sin(2 * np.pi * 4.0 * t), 0, None) ** 0.7
    # "music": a sustained chord with a beat
    music = sum(0.25 * np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63, 440.0))
    music *= 0.6 + 0.4 * (np.sin(2 * np.pi * 2.0 * t) > 0)
    noise = 0.05 * rng.standard_normal(n)
    x = 0.25 * voice + 0.15 * music + noise
    x[: SR // 20] *= np.linspace(0, 1, SR // 20)
    return x.astype(np.float32)


def run(tag):
    sess = ort.InferenceSession(os.path.join(MODELS, f'fastenhancer_{tag}.onnx'), providers=['CPUExecutionProvider'])
    ins = sess.get_inputs()
    cache_in = sorted([i for i in ins if i.name.startswith('cache_in_')], key=lambda i: int(i.name.split('_')[-1]))
    hop = ins[[i.name for i in ins].index('wav_in')].shape[1]
    caches = [np.zeros([d if isinstance(d, int) else 1 for d in c.shape], np.float32) for c in cache_in]
    outs = [o.name for o in sess.get_outputs()]
    cache_out = sorted([o for o in outs if o.startswith('cache_out_')], key=lambda s: int(s.split('_')[-1]))

    x = test_signal(HOPS * hop)
    y = np.zeros_like(x)
    for h in range(HOPS):
        feeds = {'wav_in': x[h * hop:(h + 1) * hop][None]}
        feeds.update({c.name: v for c, v in zip(cache_in, caches)})
        res = dict(zip(outs, sess.run(None, feeds)))
        y[h * hop:(h + 1) * hop] = res['wav_out'][0]
        caches = [res[n] for n in cache_out]

    os.makedirs(OUT, exist_ok=True)
    x.tofile(os.path.join(OUT, f'{tag}_in.f32'))
    y.tofile(os.path.join(OUT, f'{tag}_out.f32'))
    meta = {'hop': hop, 'hops': HOPS, 'caches': [int(c.size) for c in caches],
            'final_caches': [c.ravel().tolist() for c in caches]}
    with open(os.path.join(OUT, f'{tag}.json'), 'w') as f:
        json.dump(meta, f)
    print(f'{tag}: {HOPS} hops, out rms {float(np.sqrt(np.mean(y ** 2))):.4f}')


if __name__ == '__main__':
    for tag in ('t', 's'):
        run(tag)
