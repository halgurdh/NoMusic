#!/usr/bin/env python3
"""
fe_compile.py - ahead-of-time compiler: FastEnhancer ONNX graph -> standalone Rust/WASM code.

The streaming FastEnhancer graphs have fully static shapes ([1, 512] hop in, fixed-size caches),
so the whole network can be lowered at build time into straight-line Rust code:

  * every shape / index computation is constant-folded away,
  * Reshape / Transpose / Slice / Gather / Split / Squeeze / Unsqueeze / Expand become zero-copy
    strided *views* (they emit no code); data is only materialised when a kernel needs it,
  * Sigmoid*x is fused into SiLU and folded into the producing Conv,
  * Conv / ConvTranspose / Gemm / MatMul all lower onto one SIMD sgemm micro-kernel,
  * Gemm weights are pre-transposed, FFT twiddles are pre-computed,
  * all intermediates live in one arena whose offsets come from a liveness-based memory planner.

Output: <out>.rs (generated Rust module) + <out>.bin (float32 weight blob, embedded via include_bytes!).

Usage: python fe_compile.py model.onnx --out ../runtime/src/generated/model_s --name fastenhancer_s
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

ALIGN = 4  # floats (16 bytes)


# ----------------------------------------------------------------------------------------------
# Buffers & views
# ----------------------------------------------------------------------------------------------

class Buf:
    """A storage region. kind: 'arena' | 'w' (weights) | 'x' (hop in) | 'y' (hop out) |
    'ci' (cache in k) | 'co' (cache out k)."""

    def __init__(self, kind, size, idx=None, name='', numel=None):
        self.kind = kind
        self.size = size
        self.numel = size if numel is None else numel
        self.idx = idx
        self.name = name
        self.base = None       # arena offset (assigned by the memory planner)
        self.redirect = None   # arena buffer replaced by an output buffer

    def resolve(self):
        b = self
        while b.redirect is not None:
            b = b.redirect
        return b

    def ptr(self, off):
        b = self.resolve()
        if b.kind == 'arena':
            assert b.base is not None, 'unplanned arena buffer ' + b.name
            return f'a.add({b.base + off})'
        if b.kind == 'w':
            return f'w.add({off})'
        if b.kind == 'x':
            return f'x.add({off})'
        if b.kind == 'y':
            return f'y.add({off})'
        if b.kind == 'ci':
            return f'ci[{b.idx}].add({off})'
        if b.kind == 'co':
            return f'co[{b.idx}].add({off})'
        raise ValueError(b.kind)


def contig_strides(shape):
    st, acc = [], 1
    for d in reversed(shape):
        st.append(acc)
        acc *= d
    return tuple(reversed(st))


def prod(xs):
    p = 1
    for x in xs:
        p *= int(x)
    return p


class View:
    __slots__ = ('buf', 'off', 'shape', 'strides')

    def __init__(self, buf, off, shape, strides=None):
        self.buf = buf
        self.off = int(off)
        self.shape = tuple(int(d) for d in shape)
        self.strides = tuple(int(s) for s in (strides if strides is not None else contig_strides(self.shape)))
        assert len(self.shape) == len(self.strides)

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return prod(self.shape)

    def is_contig(self):
        exp = contig_strides(self.shape)
        return all(d == 1 or s == e for d, s, e in zip(self.shape, self.strides, exp))

    def key(self):
        return (id(self.buf), self.off, self.shape, self.strides)

    def P(self, extra=0):
        return self.buf.ptr(self.off + extra)


class Seq:
    def __init__(self, items):
        self.items = list(items)


def coalesce(shape, *strides):
    """Drop size-1 dims and merge adjacent dims that are linearly addressable for every operand."""
    dims = [(n, [s[i] for s in strides]) for i, n in enumerate(shape) if n != 1]
    if not dims:
        return [1], [[0] for _ in strides]
    merged = [dims[0]]
    for n, ss in dims[1:]:
        pn, pss = merged[-1]
        if all(ps == s * n for ps, s in zip(pss, ss)):
            merged[-1] = (pn * n, ss)
        else:
            merged.append((n, ss))
    shp = [n for n, _ in merged]
    sts = [[ss[k] for _, ss in merged] for k in range(len(strides))]
    return shp, sts


def broadcast_view(v, shape):
    shape = tuple(shape)
    pad = len(shape) - v.ndim
    assert pad >= 0
    vs = (1,) * pad + v.shape
    st = (0,) * pad + v.strides
    nst = []
    for d, vd, s in zip(shape, vs, st):
        if vd == d:
            nst.append(s if d != 1 else 0)
        elif vd == 1:
            nst.append(0)
        else:
            raise ValueError(f'cannot broadcast {v.shape} to {shape}')
    return View(v.buf, v.off, shape, nst)


def attempt_nocopy_reshape(oshape, ostrides, nshape):
    """numpy's _attempt_nocopy_reshape, in element strides."""
    od = [(d, s) for d, s in zip(oshape, ostrides) if d != 1]
    olddims = [d for d, _ in od]
    oldstrides = [s for _, s in od]
    oldnd, newnd = len(olddims), len(nshape)
    newstrides = [0] * newnd
    oi, oj, ni, nj = 0, 1, 0, 1
    while ni < newnd and oi < oldnd:
        np_ = nshape[ni]
        op = olddims[oi]
        while np_ != op:
            if np_ < op:
                if nj >= newnd:
                    return None
                np_ *= nshape[nj]
                nj += 1
            else:
                if oj >= oldnd:
                    return None
                op *= olddims[oj]
                oj += 1
        for ok in range(oi, oj - 1):
            if oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]:
                return None
        newstrides[nj - 1] = oldstrides[oj - 1]
        for nk in range(nj - 1, ni, -1):
            newstrides[nk - 1] = newstrides[nk] * nshape[nk]
        ni = nj
        nj += 1
        oi = oj
        oj += 1
    last = newstrides[ni - 1] if ni >= 1 else 1
    for nk in range(ni, newnd):
        newstrides[nk] = last
    return newstrides


def lit(v):
    v = float(np.float32(v))
    if math.isnan(v):
        return 'f32::NAN'
    if math.isinf(v):
        return 'f32::INFINITY' if v > 0 else 'f32::NEG_INFINITY'
    r = repr(v)
    if 'e' not in r and '.' not in r:
        r += '.0'
    return f'({r}_f32)'


def arr(xs):
    return '&[' + ', '.join(str(int(x)) for x in xs) + ']'


class Step:
    def __init__(self, render, reads, writes):
        self.render = render
        self.reads = list(reads)
        self.writes = list(writes)


# ----------------------------------------------------------------------------------------------
# Graph preparation
# ----------------------------------------------------------------------------------------------

def attrs_of(node):
    out = {}
    for a in node.attribute:
        v = helper.get_attribute_value(a)
        if isinstance(v, bytes):
            v = v.decode()
        out[a.name] = v
    return out


def prepare(model):
    try:
        import onnxsim
        sim, ok = onnxsim.simplify(model)
        if ok:
            model = sim
    except Exception as exc:  # onnxsim is optional
        print('warning: onnxsim unavailable or failed (%s); compiling unsimplified graph' % exc)
    return shape_inference.infer_shapes(model)


def fuse(nodes, graph_outputs):
    """Sigmoid(x)*x -> SiLU(x);  Conv -> SiLU  ->  Conv(act=silu)."""
    def consumers_map(ns):
        c = defaultdict(list)
        for n in ns:
            for i in n.input:
                if i:
                    c[i].append(n)
        return c

    cons = consumers_map(nodes)
    drop = set()
    replace = {}
    for sig in nodes:
        if sig.op_type != 'Sigmoid':
            continue
        x, s = sig.input[0], sig.output[0]
        cs = cons[s]
        if len(cs) != 1 or s in graph_outputs:
            continue
        mul = cs[0]
        if mul.op_type != 'Mul' or sorted(mul.input) != sorted([x, s]):
            continue
        drop.add(id(sig))
        replace[id(mul)] = helper.make_node('SiLU', [x], [mul.output[0]], name=mul.name + '_silu')
    nodes = [replace.get(id(n), n) for n in nodes if id(n) not in drop]

    cons = consumers_map(nodes)
    drop = set()
    replace = {}
    for conv in nodes:
        if conv.op_type != 'Conv':
            continue
        c = conv.output[0]
        cs = cons[c]
        if len(cs) != 1 or c in graph_outputs or cs[0].op_type != 'SiLU':
            continue
        act = cs[0]
        drop.add(id(act))
        a = attrs_of(conv)
        a['fe_act'] = 'silu'
        replace[id(conv)] = helper.make_node('Conv', list(conv.input), [act.output[0]], name=conv.name, **a)
    return [replace.get(id(n), n) for n in nodes if id(n) not in drop]


# ----------------------------------------------------------------------------------------------
# Compiler
# ----------------------------------------------------------------------------------------------

class Compiler:
    def __init__(self, model, winograd=True):
        self.model = model
        self.winograd = winograd
        self.g = model.graph
        self.opset = max([o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')] or [18])
        self.inferred = {}
        for vi in list(self.g.value_info) + list(self.g.input) + list(self.g.output):
            tt = vi.type.tensor_type
            if vi.type.HasField('tensor_type') and tt.HasField('shape'):
                dims = [d.dim_value if d.HasField('dim_value') else None for d in tt.shape.dim]
                if all(d is not None for d in dims):
                    self.inferred[vi.name] = tuple(dims)
        self.env = {}
        self.steps = []
        self.bufs = []
        self.wbuf = Buf('w', 0, name='weights')
        self.wchunks = []
        self.wlen = 0
        self.wcache = {}
        self.matcache = {}
        self.twcache = {}
        self.comments = []

        for init in self.g.initializer:
            self.env[init.name] = numpy_helper.to_array(init)

        init_names = {i.name for i in self.g.initializer}
        rt_inputs = [i for i in self.g.input if i.name not in init_names]
        caches_in = sorted([i for i in rt_inputs if i.name.startswith('cache_in_')],
                           key=lambda v: int(v.name.split('_')[-1]))
        audio_in = [i for i in rt_inputs if not i.name.startswith('cache_in_')]
        if len(audio_in) != 1:
            raise SystemExit('expected exactly one non-cache graph input, got %s' % [i.name for i in audio_in])
        caches_out = sorted([o for o in self.g.output if o.name.startswith('cache_out_')],
                            key=lambda v: int(v.name.split('_')[-1]))
        audio_out = [o for o in self.g.output if not o.name.startswith('cache_out_')]
        if len(audio_out) != 1:
            raise SystemExit('expected exactly one non-cache graph output')
        if [c.name[len('cache_in_'):] for c in caches_in] != [c.name[len('cache_out_'):] for c in caches_out]:
            raise SystemExit('cache_in_* / cache_out_* do not pair up')

        self.x_name = audio_in[0].name
        self.y_name = audio_out[0].name
        xshape = self.inferred[self.x_name]
        self.hop = prod(xshape)
        self.env[self.x_name] = View(Buf('x', self.hop, name=self.x_name), 0, xshape)
        self.cache_shapes = []
        for k, ci in enumerate(caches_in):
            shp = self.inferred[ci.name]
            assert self.inferred[caches_out[k].name] == shp, 'cache in/out shape mismatch'
            self.cache_shapes.append(shp)
            self.env[ci.name] = View(Buf('ci', prod(shp), idx=k, name=ci.name), 0, shp)

        yshape = self.inferred[self.y_name]
        assert prod(yshape) == self.hop
        self.out_bufs = {self.y_name: Buf('y', self.hop, name=self.y_name)}
        for k, co in enumerate(caches_out):
            self.out_bufs[co.name] = Buf('co', prod(self.cache_shapes[k]), idx=k, name=co.name)
        self.out_done = set()

    # ---------------- helpers ----------------

    def emit(self, render, reads, writes):
        self.steps.append(Step(render, [v.buf if isinstance(v, View) else v for v in reads],
                               [v.buf if isinstance(v, View) else v for v in writes]))

    def new(self, shape, name=''):
        n = max(1, prod(shape))
        b = Buf('arena', (n + ALIGN - 1) // ALIGN * ALIGN, name=name, numel=n)
        self.bufs.append(b)
        return View(b, 0, shape)

    def wview(self, a, key=None):
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
        k = key if key is not None else (a.shape, a.tobytes())
        if k in self.wcache:
            return self.wcache[k]
        off = self.wlen
        self.wchunks.append(a.ravel())
        pad = (-a.size) % ALIGN
        if pad:
            self.wchunks.append(np.zeros(pad, np.float32))
        self.wlen += a.size + pad
        v = View(self.wbuf, off, a.shape)
        self.wcache[k] = v
        return v

    def rt(self, x):
        """Runtime view for a value (constants are placed into the weight blob)."""
        if isinstance(x, View):
            return x
        if isinstance(x, np.ndarray):
            if x.dtype.kind not in 'fiub':
                raise TypeError('unsupported constant dtype %s' % x.dtype)
            return self.wview(x.astype(np.float32))
        raise TypeError(type(x))

    def contig(self, v):
        if v.is_contig():
            return v
        k = v.key()
        if k in self.matcache:
            return self.matcache[k]
        nv = self.new(v.shape, 'mat')
        self.copy(nv, v)
        self.matcache[k] = nv
        return nv

    def unit_last(self, v):
        """Ensure stride 1 along the last axis (rows may be strided)."""
        if v.shape[-1] == 1 or v.strides[-1] == 1:
            return v
        return self.contig(v)

    def copy(self, dst, src):
        src = broadcast_view(src, dst.shape)
        if dst.numel() == 0:
            return
        shp, (ds, ss) = coalesce(dst.shape, dst.strides, src.strides)
        self.emit(lambda: f'k::copy_nd({dst.P()}, {arr(ds)}, {src.P()}, {arr(ss)}, {arr(shp)});', [src], [dst])

    def fill(self, dst, value):
        assert dst.is_contig()
        n = dst.numel()
        self.emit(lambda: f'k::fill({dst.P()}, {n}, {lit(value)});', [], [dst])

    def map1(self, x, expr, shape=None):
        """out = expr(x) elementwise; expr uses the variable `x`."""
        shape = tuple(shape) if shape is not None else x.shape
        x = broadcast_view(x, shape)
        out = self.new(shape, 'map')
        shp, (ds, xs) = coalesce(shape, out.strides, x.strides)
        self.emit(lambda: f'k::map_nd({out.P()}, {x.P()}, {arr(xs)}, {arr(shp)}, |x: f32| {expr});', [x], [out])
        return out

    def act(self, x, act):
        out = self.new(x.shape, act.lower())
        shp, (ds, xs) = coalesce(x.shape, out.strides, x.strides)
        self.emit(lambda: f'k::act_nd::<k::{act}>({out.P()}, {x.P()}, {arr(xs)}, {arr(shp)});', [x], [out])
        return out

    def map2(self, a, b, expr):
        shape = tuple(np.broadcast_shapes(a.shape, b.shape))
        a = broadcast_view(a, shape)
        b = broadcast_view(b, shape)
        out = self.new(shape, 'zip')
        shp, (ds, as_, bs) = coalesce(shape, out.strides, a.strides, b.strides)
        self.emit(lambda: f'k::zip_nd({out.P()}, {a.P()}, {arr(as_)}, {b.P()}, {arr(bs)}, {arr(shp)}, '
                          f'|a: f32, b: f32| {expr});', [a, b], [out])
        return out

    def sgemm(self, m, n, kk, alpha, A, sam, sak, B, ldb, C, ldc, acc, reads, writes):
        self.emit(lambda: f'k::sgemm({m}, {n}, {kk}, {lit(alpha)}, {A.P()}, {sam}, {sak}, {B.P()}, {ldb}, '
                          f'{C.P()}, {ldc}, {"true" if acc else "false"});', reads, writes)

    def sgemm_ex(self, m, n, kk, A, sam, sak, B, ldb, C, ldc, reads, writes, act='Ident', brow=None, bcol=None,
                 alpha=1.0, acc=False):
        """GEMM with the bias/activation epilogue fused into the register tile."""
        def epi():
            if brow is not None and bcol is None and not acc:
                e = f'k::Epi::rows({brow.P()})'
            elif bcol is not None and brow is None and not acc:
                e = f'k::Epi::cols({bcol.P()})'
            else:
                e = (f'k::Epi {{ alpha: 1.0, acc: {"true" if acc else "false"}, '
                     f'brow: {brow.P() if brow is not None else "core::ptr::null()"}, '
                     f'bcol: {bcol.P() if bcol is not None else "core::ptr::null()"} }}')
            return e if alpha == 1.0 else f'{e}.alpha({lit(alpha)})'
        self.emit(lambda: f'k::sgemm_ex::<k::{act}>({m}, {n}, {kk}, {A.P()}, {sam}, {sak}, {B.P()}, {ldb}, '
                          f'{C.P()}, {ldc}, {epi()});', reads, writes)

    def reshape_view(self, v, shape):
        shape = tuple(int(d) for d in shape)
        assert prod(shape) == v.numel(), (v.shape, shape)
        if v.is_contig():
            return View(v.buf, v.off, shape)
        st = attempt_nocopy_reshape(v.shape, v.strides, shape)
        if st is not None:
            return View(v.buf, v.off, shape, st)
        c = self.contig(v)
        return View(c.buf, c.off, shape)

    def const_eval(self, node, vals):
        from onnx.reference import ReferenceEvaluator
        ins = [i for i in node.input if i]
        feeds = {}
        vi_in = []
        for name, v in zip(node.input, vals):
            if name:
                feeds[name] = v
                vi_in.append(helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(v.dtype), v.shape))
        vi_out = [helper.make_empty_tensor_value_info(o) for o in node.output]
        g = helper.make_graph([node], 'cf', vi_in, vi_out)
        m = helper.make_model(g, opset_imports=list(self.model.opset_import))
        res = ReferenceEvaluator(m).run(None, {k: feeds[k] for k in ins})
        return [np.asarray(r) for r in res]

    # ---------------- driver ----------------

    def compile(self):
        nodes = fuse(list(self.g.node), set(self.out_bufs))
        for name in self.out_bufs:
            if name in self.env:
                self.finish_output(name)
        for idx, node in enumerate(nodes):
            vals = [self.env[i] if i else None for i in node.input]
            n_before = len(self.steps)
            if all(isinstance(v, np.ndarray) or v is None for v in vals) and node.op_type not in ('SiLU',):
                outs = self.const_eval(node, vals)
            else:
                h = getattr(self, 'op_' + node.op_type, None)
                if h is None:
                    raise SystemExit(f'unsupported op {node.op_type} ({node.name})')
                outs = h(node, vals, attrs_of(node))
                if not isinstance(outs, (list, tuple)):
                    outs = [outs]
            for name, v in zip(node.output, outs):
                if not name:
                    continue
                self.env[name] = v
                if name in self.inferred and not isinstance(v, Seq):
                    got = tuple(v.shape)
                    if got != self.inferred[name]:
                        raise SystemExit(f'shape mismatch at {node.op_type} {name}: {got} vs {self.inferred[name]}')
            if len(self.steps) > n_before:
                self.steps[n_before].comment = f'{node.op_type} {node.name} -> {", ".join(node.output)}'
            for name in node.output:
                if name in self.out_bufs:
                    self.finish_output(name)
        missing = set(self.out_bufs) - self.out_done
        if missing:
            raise SystemExit('outputs never produced: %s' % missing)

    def finish_output(self, name):
        if name in self.out_done:
            return
        self.out_done.add(name)
        ob = self.out_bufs[name]
        v = self.rt(self.env[name])
        # Let the producer write straight into the output buffer when the value covers its whole buffer.
        if (v.buf.kind == 'arena' and v.buf.redirect is None and v.off == 0 and v.is_contig()
                and v.numel() == ob.size and v.buf.numel == ob.size):
            v.buf.redirect = ob
            return
        self.copy(View(ob, 0, v.shape), v)

    # ---------------- memory planning ----------------

    def plan(self):
        first, last = {}, {}
        for i, st in enumerate(self.steps):
            for b in st.reads + st.writes:
                b = b.resolve()
                if b.kind != 'arena':
                    continue
                first.setdefault(id(b), i)
                last[id(b)] = i
        live = [b for b in self.bufs if b.redirect is None and id(b) in first]
        live.sort(key=lambda b: (first[id(b)], -b.size))
        active = []  # (base, size, last)
        top = 0
        for b in live:
            t = first[id(b)]
            active = [a for a in active if a[2] >= t]
            active.sort()
            base = 0
            for (ab, asz, _) in active:
                if base + b.size <= ab:
                    break
                base = max(base, ab + asz)
            b.base = base
            active.append((base, b.size, last[id(b)]))
            top = max(top, base + b.size)
        self.arena_len = max(top, ALIGN)
        self.naive_arena = sum(b.size for b in live)

    # ---------------- rendering ----------------

    def render(self, model_name, bin_name):
        lines = []
        w = lines.append
        w('// @generated by wasm/compiler/fe_compile.py - DO NOT EDIT.')
        w(f'// Source graph: {model_name}.onnx ({len(self.g.node)} ONNX nodes -> {len(self.steps)} kernel calls)')
        w(f'// Weights: {self.wlen} f32 ({self.wlen * 4} bytes). Arena: {self.arena_len} f32 '
          f'(naive {self.naive_arena}).')
        w('#![allow(clippy::all, unused_unsafe, unused_parens)]')
        w('use crate::kernels as k;')
        w('')
        w(f'pub const MODEL_NAME: &str = "{model_name}";')
        w(f'pub const HOP: usize = {self.hop};')
        w(f'pub const ARENA_LEN: usize = {self.arena_len};')
        w(f'pub const WEIGHTS_LEN: usize = {self.wlen};')
        w(f'pub const NCACHE: usize = {len(self.cache_shapes)};')
        w('pub const CACHE_LENS: [usize; NCACHE] = [' + ', '.join(str(prod(s)) for s in self.cache_shapes) + '];')
        w(f'pub static WEIGHTS: &k::Aligned<[u8]> = &k::Aligned(*include_bytes!("{bin_name}"));')
        w('')
        w('/// One streaming hop. `x`: HOP input samples, `y`: HOP output samples,')
        w('/// `ci`/`co`: cache state in/out (distinct buffers), `a`: ARENA_LEN scratch, `w`: weights.')
        w('#[inline(never)]')
        w('pub unsafe fn run(a: *mut f32, w: *const f32, x: *const f32, y: *mut f32,')
        w('                  ci: &[*const f32; NCACHE], co: &[*mut f32; NCACHE]) {')
        w('    let _ = (a, w, x, y, ci, co);')
        w('    #[cfg(feature = "profile")] crate::prof::start();')
        labels = []
        op = '?'
        for i, st in enumerate(self.steps):
            c = getattr(st, 'comment', None)
            if c:
                w(f'    // {c}')
                op = c.split(' ')[0]
            code = st.render()
            w('    ' + code)
            w(f'    #[cfg(feature = "profile")] crate::prof::mark({i});')
            labels.append(f'{op}:{code.split("(")[0].replace("k::", "")}')
        w('}')
        w('')
        w('/// Per-kernel-call labels (op:kernel) for the `profile` feature.')
        w('#[allow(dead_code)]')
        w(f'pub const PROF_LABELS: [&str; {len(labels)}] = [' + ', '.join(f'"{s}"' for s in labels) + '];')
        return '\n'.join(lines) + '\n'

    def weights_blob(self):
        if not self.wchunks:
            return np.zeros(ALIGN, np.float32).tobytes()
        return np.concatenate(self.wchunks).astype('<f4').tobytes()

    # ==============================================================================================
    # Ops
    # ==============================================================================================

    # ---- views (no code) ----

    def op_Identity(self, n, v, a):
        return v[0]

    op_Dropout = op_Identity

    def op_Cast(self, n, v, a):
        to = a['to']
        if to != onnx.TensorProto.FLOAT:
            raise SystemExit('Cast of a runtime tensor to non-float is unsupported')
        return v[0]

    def op_Shape(self, n, v, a):
        shp = np.array(v[0].shape, dtype=np.int64)
        s, e = a.get('start', 0), a.get('end', None)
        return shp[s:e]

    def op_Size(self, n, v, a):
        return np.array(v[0].numel(), dtype=np.int64)

    def op_Reshape(self, n, v, a):
        x = v[0]
        shape = [int(d) for d in v[1]]
        allowzero = a.get('allowzero', 0)
        out = []
        for i, d in enumerate(shape):
            out.append(x.shape[i] if (d == 0 and not allowzero) else d)
        if -1 in out:
            known = prod(d for d in out if d != -1)
            out[out.index(-1)] = x.numel() // known
        return self.reshape_view(x, out)

    def op_Flatten(self, n, v, a):
        x = v[0]
        ax = a.get('axis', 1)
        ax = ax + x.ndim if ax < 0 else ax
        return self.reshape_view(x, [prod(x.shape[:ax]), prod(x.shape[ax:])])

    def op_Squeeze(self, n, v, a):
        x = v[0]
        axes = [int(t) for t in v[1]] if len(v) > 1 and v[1] is not None else a.get('axes')
        if axes is None:
            axes = [i for i, d in enumerate(x.shape) if d == 1]
        axes = {ax + x.ndim if ax < 0 else ax for ax in axes}
        keep = [i for i in range(x.ndim) if i not in axes]
        return View(x.buf, x.off, [x.shape[i] for i in keep], [x.strides[i] for i in keep])

    def op_Unsqueeze(self, n, v, a):
        x = v[0]
        axes = [int(t) for t in v[1]] if len(v) > 1 and v[1] is not None else a['axes']
        r = x.ndim + len(axes)
        axes = sorted(ax + r if ax < 0 else ax for ax in axes)
        shp, st = list(x.shape), list(x.strides)
        for ax in axes:
            shp.insert(ax, 1)
            st.insert(ax, 0)
        return View(x.buf, x.off, shp, st)

    def op_Transpose(self, n, v, a):
        x = self.rt(v[0])
        perm = a.get('perm', list(range(x.ndim))[::-1])
        return View(x.buf, x.off, [x.shape[p] for p in perm], [x.strides[p] for p in perm])

    def op_Expand(self, n, v, a):
        x = self.rt(v[0])
        shape = np.broadcast_shapes(x.shape, tuple(int(d) for d in v[1]))
        return broadcast_view(x, shape)

    def _slice(self, x, axis, start, end, step):
        dim = x.shape[axis]
        if step == 0:
            raise SystemExit('Slice step 0')
        if start < 0:
            start += dim
        if end < 0:
            end += dim
        if step > 0:
            start = min(max(start, 0), dim)
            end = min(max(end, 0), dim)
            size = max(0, -(-(end - start) // step))
        else:
            start = min(max(start, 0), dim - 1)
            end = min(max(end, -1), dim - 1)
            size = max(0, -(-(start - end) // -step))
        shp, st = list(x.shape), list(x.strides)
        off = x.off + start * st[axis] if size > 0 else x.off
        shp[axis] = size
        st[axis] = st[axis] * step
        return View(x.buf, off, shp, st)

    def op_Slice(self, n, v, a):
        x = self.rt(v[0])
        starts = [int(t) for t in v[1]]
        ends = [int(t) for t in v[2]]
        axes = [int(t) for t in v[3]] if len(v) > 3 and v[3] is not None else list(range(len(starts)))
        steps = [int(t) for t in v[4]] if len(v) > 4 and v[4] is not None else [1] * len(starts)
        for s, e, ax, sp in zip(starts, ends, axes, steps):
            ax = ax + x.ndim if ax < 0 else ax
            x = self._slice(x, ax, s, e, sp)
        return x

    def op_Gather(self, n, v, a):
        x = self.rt(v[0])
        idx = np.asarray(v[1])
        if not isinstance(v[1], np.ndarray):
            raise SystemExit('Gather with runtime indices is unsupported')
        ax = a.get('axis', 0)
        ax = ax + x.ndim if ax < 0 else ax
        dim = x.shape[ax]
        idx = np.where(idx < 0, idx + dim, idx).astype(np.int64)
        shp, st = list(x.shape), list(x.strides)
        if idx.ndim == 0:
            return View(x.buf, x.off + int(idx) * st[ax], shp[:ax] + shp[ax + 1:], st[:ax] + st[ax + 1:])
        flat = idx.ravel()
        if idx.ndim == 1 and (flat.size == 1 or np.all(np.diff(flat) == flat[1] - flat[0])):
            step = int(flat[1] - flat[0]) if flat.size > 1 else 1
            return View(x.buf, x.off + int(flat[0]) * st[ax], shp[:ax] + [flat.size] + shp[ax + 1:],
                        st[:ax] + [st[ax] * step] + st[ax + 1:])
        if flat.size > 4096:
            raise SystemExit('Gather with too many arbitrary indices')
        out = self.new(shp[:ax] + list(idx.shape) + shp[ax + 1:], 'gather')
        o3 = View(out.buf, 0, [prod(shp[:ax]), flat.size, prod(shp[ax + 1:])])
        for j, i in enumerate(flat):
            src = View(x.buf, x.off + int(i) * st[ax], shp[:ax] + shp[ax + 1:], st[:ax] + st[ax + 1:])
            dst = View(out.buf, j * o3.strides[1], shp[:ax] + shp[ax + 1:],
                       list(contig_strides(out.shape)[:ax]) + list(contig_strides(out.shape)[ax + idx.ndim:]))
            self.copy(dst, src)
        return out

    def _split_sizes(self, dim, v, a, nout):
        if len(v) > 1 and v[1] is not None:
            return [int(t) for t in v[1]]
        if 'split' in a:
            return list(a['split'])
        num = a.get('num_outputs', nout)
        chunk = -(-dim // num)
        sizes = [chunk] * (num - 1) + [dim - chunk * (num - 1)]
        return sizes

    def op_Split(self, n, v, a):
        x = self.rt(v[0])
        ax = a.get('axis', 0)
        ax = ax + x.ndim if ax < 0 else ax
        sizes = self._split_sizes(x.shape[ax], v, a, len(n.output))
        outs, pos = [], 0
        for s in sizes:
            outs.append(self._slice(x, ax, pos, pos + s, 1))
            pos += s
        return outs

    def op_SplitToSequence(self, n, v, a):
        x = self.rt(v[0])
        ax = a.get('axis', 0)
        ax = ax + x.ndim if ax < 0 else ax
        keep = a.get('keepdims', 1)
        if len(v) > 1 and v[1] is not None:
            sp = np.asarray(v[1])
            sizes = [int(sp)] * (-(-x.shape[ax] // int(sp))) if sp.ndim == 0 else [int(t) for t in sp]
            if sp.ndim == 0:
                sizes[-1] = x.shape[ax] - int(sp) * (len(sizes) - 1)
            items, pos = [], 0
            for s in sizes:
                items.append(self._slice(x, ax, pos, pos + s, 1))
                pos += s
            return Seq(items)
        items = []
        for i in range(x.shape[ax]):
            sl = self._slice(x, ax, i, i + 1, 1)
            if not keep:
                sl = View(sl.buf, sl.off, sl.shape[:ax] + sl.shape[ax + 1:], sl.strides[:ax] + sl.strides[ax + 1:])
            items.append(sl)
        return Seq(items)

    def op_SequenceAt(self, n, v, a):
        seq = v[0]
        i = int(np.asarray(v[1]))
        return seq.items[i]

    # ---- data movement ----

    def op_Concat(self, n, v, a):
        xs = [self.rt(t) for t in v if t is not None]
        ax = a['axis']
        ax = ax + xs[0].ndim if ax < 0 else ax
        shape = list(xs[0].shape)
        shape[ax] = sum(t.shape[ax] for t in xs)
        out = self.new(shape, 'concat')
        pos = 0
        for t in xs:
            if t.numel() == 0:
                continue
            dst = self._slice(out, ax, pos, pos + t.shape[ax], 1)
            self.copy(dst, t)
            pos += t.shape[ax]
        return out

    def op_Pad(self, n, v, a):
        x = self.rt(v[0])
        mode = a.get('mode', 'constant')
        if mode != 'constant':
            raise SystemExit('Pad mode %s unsupported' % mode)
        pads = [int(t) for t in (v[1] if len(v) > 1 and v[1] is not None else a['pads'])]
        cval = float(np.asarray(v[2]).ravel()[0]) if len(v) > 2 and v[2] is not None and np.asarray(v[2]).size else 0.0
        axes = [int(t) for t in v[3]] if len(v) > 3 and v[3] is not None else list(range(x.ndim))
        axes = [ax + x.ndim if ax < 0 else ax for ax in axes]
        r = len(axes)
        before = [0] * x.ndim
        after = [0] * x.ndim
        for i, ax in enumerate(axes):
            before[ax] = pads[i]
            after[ax] = pads[i + r]
        for ax in range(x.ndim):  # negative pads crop
            if before[ax] < 0:
                x = self._slice(x, ax, -before[ax], x.shape[ax], 1)
                before[ax] = 0
            if after[ax] < 0:
                x = self._slice(x, ax, 0, x.shape[ax] + after[ax], 1)
                after[ax] = 0
        shape = [d + b + e for d, b, e in zip(x.shape, before, after)]
        out = self.new(shape, 'pad')
        if any(before) or any(after):
            self.fill(out, cval)
        dst = out
        for ax in range(x.ndim):
            dst = self._slice(dst, ax, before[ax], before[ax] + x.shape[ax], 1)
        self.copy(dst, x)
        return out

    def op_ScatterND(self, n, v, a):
        data = self.rt(v[0])
        if not isinstance(v[1], np.ndarray):
            raise SystemExit('ScatterND with runtime indices unsupported')
        if a.get('reduction', 'none') != 'none':
            raise SystemExit('ScatterND reduction unsupported')
        idx = np.asarray(v[1]).astype(np.int64)
        upd = self.rt(v[2])
        out = self.new(data.shape, 'scatter')
        self.copy(out, data)
        q = idx.shape[-1]
        rows = idx.reshape(-1, q)
        inner = list(data.shape[q:])
        isz = prod(inner)
        ost = contig_strides(data.shape)
        offs = [sum(int((r[i] + data.shape[i]) % data.shape[i]) * ost[i] for i in range(q)) for r in rows]
        u2 = self.reshape_view(upd, [len(offs)] + inner)
        # merge consecutive runs into single strided copies
        i = 0
        while i < len(offs):
            j = i + 1
            while j < len(offs) and offs[j] == offs[j - 1] + isz:
                j += 1
            cnt = j - i
            dst = View(out.buf, offs[i], [cnt] + inner, [isz] + list(contig_strides(inner)))
            src = View(u2.buf, u2.off + i * u2.strides[0], [cnt] + inner, list(u2.strides))
            self.copy(dst, src)
            i = j
        return out

    # ---- elementwise ----

    def _binary(self, v, expr_ab, expr_xs=None, expr_sx=None):
        A, B = v[0], v[1]
        if isinstance(B, np.ndarray) and B.size == 1 and expr_xs:
            xa = self.rt(A)
            shape = np.broadcast_shapes(xa.shape, B.shape)
            return self.map1(xa, expr_xs(float(B.ravel()[0])), shape)
        if isinstance(A, np.ndarray) and A.size == 1 and expr_sx:
            xb = self.rt(B)
            shape = np.broadcast_shapes(xb.shape, A.shape)
            return self.map1(xb, expr_sx(float(A.ravel()[0])), shape)
        return self.map2(self.rt(A), self.rt(B), expr_ab)

    def op_Add(self, n, v, a):
        return self._binary(v, 'a + b', lambda s: f'x + {lit(s)}', lambda s: f'{lit(s)} + x')

    def op_Sub(self, n, v, a):
        return self._binary(v, 'a - b', lambda s: f'x - {lit(s)}', lambda s: f'{lit(s)} - x')

    def op_Mul(self, n, v, a):
        return self._binary(v, 'a * b', lambda s: f'x * {lit(s)}', lambda s: f'{lit(s)} * x')

    def op_Div(self, n, v, a):
        return self._binary(v, 'a / b', lambda s: f'x * {lit(1.0 / s)}' if s != 0 and math.log2(abs(s)).is_integer()
                            else f'x / {lit(s)}', lambda s: f'{lit(s)} / x')

    def op_Max(self, n, v, a):
        if len(v) != 2:
            raise SystemExit('variadic Max unsupported')
        return self._binary(v, 'a.max(b)', lambda s: f'x.max({lit(s)})', lambda s: f'x.max({lit(s)})')

    def op_Min(self, n, v, a):
        if len(v) != 2:
            raise SystemExit('variadic Min unsupported')
        return self._binary(v, 'a.min(b)', lambda s: f'x.min({lit(s)})', lambda s: f'x.min({lit(s)})')

    def op_Pow(self, n, v, a):
        def by_scalar(e):
            e32 = float(np.float32(e))
            table = {1.0: 'x', 2.0: 'x * x', 3.0: 'x * x * x', 0.5: 'x.sqrt()', -1.0: '1.0 / x',
                     -0.5: '1.0 / x.sqrt()', 0.0: '1.0'}
            return table.get(e32, f'x.powf({lit(e32)})')
        X, E = v[0], v[1]
        if isinstance(E, np.ndarray) and E.size == 1 and not isinstance(X, np.ndarray):
            e32 = float(np.float32(E.ravel()[0]))
            if not e32.is_integer() and abs(e32) != 0.5:
                # Non-integer exponent: the base must be >= 0 for a real result, so use SIMD exp(e*log(x)).
                x = self.rt(X)
                shape = np.broadcast_shapes(x.shape, E.shape)
                xb = broadcast_view(x, shape)
                out = self.new(shape, 'pow')
                shp, (ds, xs) = coalesce(shape, out.strides, xb.strides)
                self.emit(lambda: f'k::pow_nd({out.P()}, {xb.P()}, {arr(xs)}, {arr(shp)}, {lit(e32)});', [xb], [out])
                return out
        return self._binary(v, 'a.powf(b)', by_scalar, lambda s: f'{lit(s)}.powf(x)')

    def op_Sigmoid(self, n, v, a):
        return self.act(self.rt(v[0]), 'Sigmoid')

    def op_Tanh(self, n, v, a):
        return self.act(self.rt(v[0]), 'Tanh')

    def op_Exp(self, n, v, a):
        return self.act(self.rt(v[0]), 'Exp')

    def op_SiLU(self, n, v, a):
        return self.act(self.rt(v[0]), 'Silu')

    def op_Relu(self, n, v, a):
        return self.map1(self.rt(v[0]), 'x.max(0.0)')

    def op_Neg(self, n, v, a):
        return self.map1(self.rt(v[0]), '-x')

    def op_Abs(self, n, v, a):
        return self.map1(self.rt(v[0]), 'x.abs()')

    def op_Sqrt(self, n, v, a):
        return self.map1(self.rt(v[0]), 'x.sqrt()')

    def op_Reciprocal(self, n, v, a):
        return self.map1(self.rt(v[0]), '1.0 / x')

    def op_Log(self, n, v, a):
        return self.map1(self.rt(v[0]), 'x.ln()')

    def op_LeakyRelu(self, n, v, a):
        al = a.get('alpha', 0.01)
        return self.map1(self.rt(v[0]), f'if x < 0.0 {{ x * {lit(al)} }} else {{ x }}')

    def op_Clip(self, n, v, a):
        if self.opset >= 11:
            lo = float(np.asarray(v[1]).ravel()[0]) if len(v) > 1 and v[1] is not None else -math.inf
            hi = float(np.asarray(v[2]).ravel()[0]) if len(v) > 2 and v[2] is not None else math.inf
            if (len(v) > 1 and v[1] is not None and not isinstance(v[1], np.ndarray)) or \
               (len(v) > 2 and v[2] is not None and not isinstance(v[2], np.ndarray)):
                raise SystemExit('Clip with runtime bounds unsupported')
        else:
            lo, hi = a.get('min', -math.inf), a.get('max', math.inf)
        expr = 'x'
        if lo != -math.inf:
            expr = f'{expr}.max({lit(lo)})'
        if hi != math.inf:
            expr = f'{expr}.min({lit(hi)})'
        return self.map1(self.rt(v[0]), expr)

    # ---- reductions / normalisation ----

    def op_Softmax(self, n, v, a):
        x = self.contig(self.rt(v[0]))
        ax = a.get('axis', -1)
        ax = ax + x.ndim if ax < 0 else ax
        outer, nn, inner = prod(x.shape[:ax]), x.shape[ax], prod(x.shape[ax + 1:])
        out = self.new(x.shape, 'softmax')
        self.emit(lambda: f'k::softmax({out.P()}, {x.P()}, {outer}, {nn}, {inner});', [x], [out])
        return out

    def _reduce(self, n, v, a, mode):
        x = self.rt(v[0])
        if len(v) > 1 and v[1] is not None:
            axes = [int(t) for t in v[1]]
        else:
            axes = a.get('axes')
        if not axes:
            if a.get('noop_with_empty_axes', 0):
                return x
            axes = list(range(x.ndim))
        axes = sorted({ax + x.ndim if ax < 0 else ax for ax in axes})
        keep = a.get('keepdims', 1)
        out_shape = [1 if i in axes else d for i, d in enumerate(x.shape)] if keep else \
            [d for i, d in enumerate(x.shape) if i not in axes]
        if axes == list(range(axes[0], axes[-1] + 1)):
            xc = self.contig(x)
            outer, red, inner = prod(x.shape[:axes[0]]), prod(x.shape[axes[0]:axes[-1] + 1]), prod(x.shape[axes[-1] + 1:])
        else:
            rest = [i for i in range(x.ndim) if i not in axes]
            perm = rest + axes
            xc = self.contig(View(x.buf, x.off, [x.shape[p] for p in perm], [x.strides[p] for p in perm]))
            outer, red, inner = prod(x.shape[i] for i in rest), prod(x.shape[i] for i in axes), 1
        out = self.new(out_shape, 'reduce')
        self.emit(lambda: f'k::reduce({out.P()}, {xc.P()}, {outer}, {red}, {inner}, k::Reduce::{mode});', [xc], [out])
        return out

    def op_ReduceL2(self, n, v, a):
        return self._reduce(n, v, a, 'L2')

    def op_ReduceSum(self, n, v, a):
        return self._reduce(n, v, a, 'Sum')

    def op_ReduceMean(self, n, v, a):
        return self._reduce(n, v, a, 'Mean')

    def op_ReduceMax(self, n, v, a):
        return self._reduce(n, v, a, 'Max')

    def op_ReduceMin(self, n, v, a):
        return self._reduce(n, v, a, 'Min')

    def op_ReduceSumSquare(self, n, v, a):
        return self._reduce(n, v, a, 'SumSquare')

    def op_ReduceL1(self, n, v, a):
        return self._reduce(n, v, a, 'L1')

    # ---- linear algebra ----

    def op_Gemm(self, n, v, a):
        A = self.rt(v[0])
        if a.get('transA', 0):
            A = View(A.buf, A.off, A.shape[::-1], A.strides[::-1])
        alpha, beta = a.get('alpha', 1.0), a.get('beta', 1.0)
        if isinstance(v[1], np.ndarray):
            Bm = v[1].T if a.get('transB', 0) else v[1]
            B = self.wview(np.ascontiguousarray(Bm))
        else:
            B = self.rt(v[1])
            if a.get('transB', 0):
                B = View(B.buf, B.off, B.shape[::-1], B.strides[::-1])
            B = self.unit_last(B)
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        out = self.new([M, N], 'gemm')
        ldb = B.strides[0] if K > 1 else N
        Cv = v[2] if len(v) > 2 and v[2] is not None and beta != 0.0 else None
        if Cv is None:
            self.sgemm_ex(M, N, K, A, A.strides[0], A.strides[1], B, ldb, out, N, [A, B], [out], alpha=alpha)
            return out
        if isinstance(Cv, np.ndarray):
            Cc = np.asarray(Cv, np.float32) * np.float32(beta)
            cshape = (1,) * (2 - Cc.ndim) + Cc.shape
            if cshape == (1, N):
                bias = self.wview(Cc.reshape(N))
                self.sgemm_ex(M, N, K, A, A.strides[0], A.strides[1], B, ldb, out, N, [A, B], [out],
                              bcol=bias, alpha=alpha)
                return out
            if cshape == (M, 1):
                bias = self.wview(Cc.reshape(M))
                self.sgemm_ex(M, N, K, A, A.strides[0], A.strides[1], B, ldb, out, N, [A, B], [out],
                              brow=bias, alpha=alpha)
                return out
        C = self.rt(Cv)
        if beta != 1.0:
            C = self.map1(C, f'x * {lit(beta)}')
        self.copy(out, C)
        self.sgemm_ex(M, N, K, A, A.strides[0], A.strides[1], B, ldb, out, N, [A, B, out], [out],
                      alpha=alpha, acc=True)
        return out

    def op_MatMul(self, n, v, a):
        A, B = self.rt(v[0]), self.rt(v[1])
        sq_a = sq_b = False
        if A.ndim == 1:
            A = View(A.buf, A.off, (1,) + A.shape, (0,) + A.strides)
            sq_a = True
        if B.ndim == 1:
            B = View(B.buf, B.off, B.shape + (1,), B.strides + (0,))
            sq_b = True
        B = self.unit_last(B)
        M, K = A.shape[-2:]
        K2, N = B.shape[-2:]
        assert K == K2, (A.shape, B.shape)
        batch = tuple(np.broadcast_shapes(A.shape[:-2], B.shape[:-2]))
        out = self.new(batch + (M, N), 'matmul')
        Ab = broadcast_view(View(A.buf, A.off, A.shape[:-2], A.strides[:-2]), batch) if batch else None
        Bb = broadcast_view(View(B.buf, B.off, B.shape[:-2], B.strides[:-2]), batch) if batch else None
        sam, sak = A.strides[-2], A.strides[-1]
        ldb = B.strides[-2] if K > 1 else N
        if prod(batch) <= 1:
            self.sgemm(M, N, K, 1.0, A, sam, sak, B, ldb, out, N, False, [A, B], [out])
        else:
            shp, (os_, as_, bs) = coalesce(batch, contig_strides(batch), Ab.strides, Bb.strides)
            assert os_ == list(contig_strides(shp)) or len(shp) == 1
            self.emit(lambda: f'k::matmul_batched({out.P()}, {A.P()}, {arr(as_)}, {sam}, {sak}, {B.P()}, {arr(bs)}, '
                              f'{ldb}, {arr(shp)}, {M}, {N}, {K});', [A, B], [out])
        res = out
        if sq_a or sq_b:
            shp = list(out.shape)
            if sq_b:
                shp.pop(-1)
            if sq_a:
                shp.pop(-2 if not sq_b else -1)
            res = View(out.buf, 0, shp)
        return res

    def op_Conv(self, n, v, a):
        x = self.rt(v[0])
        W = np.asarray(v[1], dtype=np.float32)
        if not isinstance(v[1], np.ndarray):
            raise SystemExit('Conv with runtime weights unsupported')
        bias = np.asarray(v[2], dtype=np.float32) if len(v) > 2 and v[2] is not None else None
        if x.ndim != 3:
            raise SystemExit('only 1-D Conv is supported (got input rank %d)' % x.ndim)
        Nb, Cin, L = x.shape
        Cout, Cg, K = W.shape
        g = a.get('group', 1)
        s = a.get('strides', [1])[0]
        d = a.get('dilations', [1])[0]
        pads = a.get('pads', [0, 0])
        if a.get('auto_pad', 'NOTSET') not in ('NOTSET', 'VALID'):
            raise SystemExit('Conv auto_pad SAME_* unsupported')
        pl, pr = pads[0], pads[1]
        Lout = (L + pl + pr - d * (K - 1) - 1) // s + 1
        assert Cg * g == Cin
        Cog = Cout // g
        x = self.unit_last(x)
        out = self.new([Nb, Cout, Lout], 'conv')
        act = 'Silu' if a.get('fe_act') == 'silu' else 'Ident'
        Bv = self.wview(bias) if bias is not None else None
        if self.winograd and K == 3 and s == 1 and d == 1 and pl == 1 and pr == 1 and L % 8 == 0 and Lout == L:
            self._conv_wino23(x, W, Bv, out, Nb, g, Cg, Cog, L, act)
            return out
        Wv = self.wview(W.reshape(Cout, Cg * K))
        for b in range(Nb):
            for gi in range(g):
                xg = View(x.buf, x.off + b * x.strides[0] + gi * Cg * x.strides[1], [Cg, L], [x.strides[1], 1])
                if K == 1 and s == 1 and pl == 0 and pr == 0:
                    Bm, ldb = xg, xg.strides[0]
                else:
                    col = self.new([Cg * K, Lout], 'im2col')
                    self.emit((lambda col=col, xg=xg: f'k::im2col_1d({col.P()}, {xg.P()}, {Cg}, {xg.strides[0]}, {L}, '
                                                      f'{K}, {s}, {d}, {pl}, {Lout});'), [xg], [col])
                    Bm, ldb = col, Lout
                Cm = View(out.buf, b * Cout * Lout + gi * Cog * Lout, [Cog, Lout])
                Am = View(self.wbuf, Wv.off + gi * Cog * Cg * K, [Cog, Cg * K])
                bb = View(self.wbuf, Bv.off + gi * Cog, [Cog]) if Bv is not None else None
                self.sgemm_ex(Cog, Lout, Cg * K, Am, Cg * K, 1, Bm, ldb, Cm, Lout, [Bm], [Cm], act=act, brow=bb)
        return out

    def _conv_wino23(self, x, W, Bv, out, Nb, g, Cg, Cog, L, act):
        """K=3, stride 1, pad 1 conv as Winograd F(2,3): 4 GEMMs on half-length inputs (see kernels.rs)."""
        h = L // 2
        Cout = Cog * g
        W64 = W.astype(np.float64)
        g0, g1, g2 = W64[:, :, 0], W64[:, :, 1], W64[:, :, 2]          # [Cout, Cg]
        U = [g0, (g0 + g1 + g2) / 2, (g0 - g1 + g2) / 2, g2]
        Uv = [self.wview(u.astype(np.float32)) for u in U]
        for b in range(Nb):
            for gi in range(g):
                xg = View(x.buf, x.off + b * x.strides[0] + gi * Cg * x.strides[1], [Cg, L], [x.strides[1], 1])
                V = self.new([4, Cg, h], 'wino_v')
                M = self.new([4, Cog, h], 'wino_m')
                self.emit((lambda V=V, xg=xg: f'k::wino23_in({V.P()}, {xg.P()}, {Cg}, {xg.strides[0]}, {L});'),
                          [xg], [V])
                for i in range(4):
                    Ai = View(self.wbuf, Uv[i].off + gi * Cog * Cg, [Cog, Cg])
                    Vi = View(V.buf, V.off + i * Cg * h, [Cg, h])
                    Mi = View(M.buf, M.off + i * Cog * h, [Cog, h])
                    self.sgemm_ex(Cog, h, Cg, Ai, Cg, 1, Vi, h, Mi, h, [V], [M])
                Cm = View(out.buf, b * Cout * L + gi * Cog * L, [Cog, L])
                bb = View(self.wbuf, Bv.off + gi * Cog, [Cog]) if Bv is not None else None
                self.emit((lambda Cm=Cm, M=M, bb=bb: f'k::wino23_out::<k::{act}>({Cm.P()}, {M.P()}, {Cog}, {L}, '
                                                     f'{bb.P() if bb is not None else "core::ptr::null()"});'),
                          [M], [Cm])

    def op_ConvTranspose(self, n, v, a):
        x = self.rt(v[0])
        if not isinstance(v[1], np.ndarray):
            raise SystemExit('ConvTranspose with runtime weights unsupported')
        W = np.asarray(v[1], dtype=np.float32)
        bias = np.asarray(v[2], dtype=np.float32) if len(v) > 2 and v[2] is not None else None
        if x.ndim != 3:
            raise SystemExit('only 1-D ConvTranspose is supported')
        if 'output_shape' in a:
            raise SystemExit('ConvTranspose output_shape unsupported')
        Nb, Cin, L = x.shape
        _, Cog, K = W.shape
        g = a.get('group', 1)
        s = a.get('strides', [1])[0]
        d = a.get('dilations', [1])[0]
        pl, pr = a.get('pads', [0, 0])
        op = a.get('output_padding', [0])[0]
        Cin_g = Cin // g
        Cout = Cog * g
        Lout = s * (L - 1) + op + ((K - 1) * d + 1) - pl - pr
        x = self.unit_last(x)
        out = self.new([Nb, Cout, Lout], 'convT')
        for b in range(Nb):
            for gi in range(g):
                Wg = W[gi * Cin_g:(gi + 1) * Cin_g]                    # [Cin_g, Cog, K]
                Wt = np.ascontiguousarray(Wg.transpose(1, 2, 0).reshape(Cog * K, Cin_g))
                Am = self.wview(Wt)
                xg = View(x.buf, x.off + b * x.strides[0] + gi * Cin_g * x.strides[1], [Cin_g, L], [x.strides[1], 1])
                col = self.new([Cog * K, L], 'col')
                self.sgemm(Cog * K, L, Cin_g, 1.0, Am, Cin_g, 1, xg, xg.strides[0], col, L, False, [xg], [col])
                Cm = View(out.buf, b * Cout * Lout + gi * Cog * Lout, [Cog, Lout])
                if bias is not None:
                    bb = self.wview(bias[gi * Cog:(gi + 1) * Cog])
                    self.emit((lambda Cm=Cm, bb=bb: f'k::bias_rows({Cm.P()}, {bb.P()}, {Cog}, {Lout});'), [], [Cm])
                else:
                    self.emit((lambda Cm=Cm: f'k::fill({Cm.P()}, {Cog * Lout}, 0.0);'), [], [Cm])
                self.emit((lambda Cm=Cm, col=col: f'k::col2im_1d({Cm.P()}, {col.P()}, {Cog}, {K}, {L}, {s}, {d}, '
                                                  f'{pl}, {Lout});'), [col, Cm], [Cm])
        return out

    # ---- spectral ----

    def op_DFT(self, n, v, a):
        x = self.rt(v[0])
        if len(v) > 1 and v[1] is not None and np.asarray(v[1]).size:
            dl = int(np.asarray(v[1]).ravel()[0])
        else:
            dl = None
        if self.opset >= 20:
            ax = int(np.asarray(v[2]).ravel()[0]) if len(v) > 2 and v[2] is not None else -2
        else:
            ax = a.get('axis', 1)
        ax = ax + x.ndim if ax < 0 else ax
        inverse = bool(a.get('inverse', 0))
        onesided = bool(a.get('onesided', 0))
        if inverse and onesided:
            raise SystemExit('DFT inverse+onesided unsupported')
        c = x.shape[-1]
        assert c in (1, 2) and ax < x.ndim - 1
        nfft = x.shape[ax]
        if dl is not None and dl != nfft:
            raise SystemExit('DFT dft_length != axis length unsupported')
        # move the transform axis next to the complex axis
        perm = [i for i in range(x.ndim - 1) if i != ax] + [ax, x.ndim - 1]
        xp = self.contig(View(x.buf, x.off, [x.shape[p] for p in perm], [x.strides[p] for p in perm]))
        outer = prod(xp.shape[:-2])
        nout = nfft // 2 + 1 if onesided else nfft
        outp = self.new(list(xp.shape[:-2]) + [nout, 2], 'dft')
        tw = self.twiddles(nfft)
        scratch = self.new([4 * nfft], 'fft_scratch')
        self.emit(lambda: f'k::dft({outp.P()}, {xp.P()}, {outer}, {nfft}, {c}, {"true" if inverse else "false"}, '
                          f'{"true" if onesided else "false"}, {tw.P()}, {scratch.P()});', [xp], [outp, scratch])
        # undo the permutation as a view
        inv = [0] * x.ndim
        for i, p in enumerate(perm):
            inv[p] = i
        return View(outp.buf, 0, [outp.shape[inv[i]] for i in range(x.ndim)],
                    [outp.strides[inv[i]] for i in range(x.ndim)])

    def twiddles(self, nfft):
        if nfft not in self.twcache:
            kk = np.arange(nfft, dtype=np.float64)
            ang = -2.0 * np.pi * kk / nfft
            tw = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
            self.twcache[nfft] = self.wview(tw, key=('tw', nfft))
        return self.twcache[nfft]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('onnx')
    ap.add_argument('--out', required=True, help='output path without extension (.rs and .bin are written)')
    ap.add_argument('--name', default=None)
    ap.add_argument('--no-winograd', action='store_true', help='lower K=3 convs via im2col + GEMM instead')
    args = ap.parse_args()

    model = prepare(onnx.load(args.onnx))
    name = args.name or os.path.splitext(os.path.basename(args.onnx))[0]
    c = Compiler(model, winograd=not args.no_winograd)
    c.compile()
    c.plan()
    bin_name = os.path.basename(args.out) + '.bin'
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + '.rs', 'w', encoding='utf-8', newline='\n') as f:
        f.write(c.render(name, bin_name))
    with open(args.out + '.bin', 'wb') as f:
        f.write(c.weights_blob())
    print(f'{name}: {len(model.graph.node)} nodes -> {len(c.steps)} kernel calls, weights {c.wlen * 4} B, '
          f'arena {c.arena_len * 4} B (unplanned {c.naive_arena * 4} B), caches {c.cache_shapes}')


if __name__ == '__main__':
    main()
