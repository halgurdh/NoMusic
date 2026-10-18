#!/usr/bin/env python3
"""Build the standalone FastEnhancer WASM modules.

  1. fe_compile.py lowers wasm/models/fastenhancer_{t,s}.onnx -> runtime/src/generated/model_{t,s}.{rs,bin}
  2. cargo builds one SIMD128 wasm32 cdylib per model (feature model_t / model_s): fastenhancer_X.wasm
  3. results are copied to assets/

Requirements: python (onnx, numpy, optional onnxsim), rustup target wasm32-unknown-unknown.
Usage: python wasm/build.py [--skip-codegen]
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNTIME = os.path.join(HERE, 'runtime')
MODELS = {'t': 'fastenhancer_t', 's': 'fastenhancer_s'}
RUSTFLAGS = '-C target-feature=+simd128'


def run(cmd, cwd=None, env=None):
    print('>', ' '.join(cmd))
    subprocess.check_call(cmd, cwd=cwd, env=env)


def main():
    skip_codegen = '--skip-codegen' in sys.argv
    for tag, name in MODELS.items():
        if not skip_codegen:
            run([sys.executable, os.path.join(HERE, 'compiler', 'fe_compile.py'),
                 os.path.join(HERE, 'models', name + '.onnx'),
                 '--out', os.path.join(RUNTIME, 'src', 'generated', 'model_' + tag), '--name', name])
        target_dir = os.path.join(RUNTIME, 'target', 'baseline')
        env = dict(os.environ, RUSTFLAGS=RUSTFLAGS, CARGO_TARGET_DIR=target_dir)
        run(['cargo', 'build', '--release', '--target', 'wasm32-unknown-unknown', '--features', 'model_' + tag],
            cwd=RUNTIME, env=env)
        src = os.path.join(target_dir, 'wasm32-unknown-unknown', 'release', 'fastenhancer_wasm.wasm')
        dst = os.path.join(ROOT, 'assets', name + '.wasm')
        shutil.copyfile(src, dst)
        print(f'  -> {os.path.relpath(dst, ROOT)} ({os.path.getsize(dst)} bytes)')


if __name__ == '__main__':
    main()
