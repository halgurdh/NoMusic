//! Hand-written compute kernels targeted by the generated model code.
//!
//! All kernels take raw pointers + static sizes/strides baked in by `fe_compile.py`.
//! On wasm32 with `+simd128` the hot paths use 128-bit SIMD; elsewhere a portable
//! lane-wise fallback with identical numerics is used (handy for native debugging).

#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use core::ptr;

/// Forces 16-byte alignment of the embedded weight blob.
#[repr(C, align(16))]
pub struct Aligned<T: ?Sized>(pub T);

// -------------------------------------------------------------------------------------------------
// 4-lane f32 vector
// -------------------------------------------------------------------------------------------------

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    pub struct F4(pub v128);

    impl F4 {
        #[inline(always)]
        pub fn splat(x: f32) -> F4 { F4(f32x4_splat(x)) }
        #[inline(always)]
        pub unsafe fn load(p: *const f32) -> F4 { F4(v128_load(p as *const v128)) }
        #[inline(always)]
        pub unsafe fn store(self, p: *mut f32) { v128_store(p as *mut v128, self.0) }
        #[inline(always)]
        pub fn add(self, o: F4) -> F4 { F4(f32x4_add(self.0, o.0)) }
        #[inline(always)]
        pub fn sub(self, o: F4) -> F4 { F4(f32x4_sub(self.0, o.0)) }
        #[inline(always)]
        pub fn mul(self, o: F4) -> F4 { F4(f32x4_mul(self.0, o.0)) }
        #[inline(always)]
        pub fn div(self, o: F4) -> F4 { F4(f32x4_div(self.0, o.0)) }
        #[inline(always)]
        pub fn min(self, o: F4) -> F4 { F4(f32x4_pmin(self.0, o.0)) }
        #[inline(always)]
        pub fn max(self, o: F4) -> F4 { F4(f32x4_pmax(self.0, o.0)) }
        #[inline(always)]
        pub fn abs(self) -> F4 { F4(f32x4_abs(self.0)) }
        #[inline(always)]
        pub fn nearest(self) -> F4 { F4(f32x4_nearest(self.0)) }
        /// 2^n for integral-valued lanes in [-126, 127].
        #[inline(always)]
        pub fn pow2i(self) -> F4 {
            let i = i32x4_trunc_sat_f32x4(self.0);
            F4(i32x4_shl(i32x4_add(i, i32x4_splat(127)), 23))
        }
        /// Copy the sign bit of `s` onto `self` (assumed non-negative).
        #[inline(always)]
        pub fn with_sign_of(self, s: F4) -> F4 {
            F4(v128_or(self.0, v128_and(s.0, u32x4_splat(0x8000_0000))))
        }
        #[inline(always)]
        pub fn lane0(self) -> f32 { f32x4_extract_lane::<0>(self.0) }
        #[inline(always)]
        pub fn hsum(self) -> f32 {
            (f32x4_extract_lane::<0>(self.0) + f32x4_extract_lane::<1>(self.0))
                + (f32x4_extract_lane::<2>(self.0) + f32x4_extract_lane::<3>(self.0))
        }
        #[inline(always)]
        pub fn hmax(self) -> f32 {
            f32x4_extract_lane::<0>(self.0).max(f32x4_extract_lane::<1>(self.0))
                .max(f32x4_extract_lane::<2>(self.0).max(f32x4_extract_lane::<3>(self.0)))
        }
        /// self + a*b
        #[inline(always)]
        pub fn madd(self, a: F4, b: F4) -> F4 {
            F4(f32x4_add(self.0, f32x4_mul(a.0, b.0)))
        }
        /// Lane shuffle over the concatenation [a0..a3, b0..b3].
        #[inline(always)]
        pub fn shuf<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(a: F4, b: F4) -> F4 {
            F4(i32x4_shuffle::<I0, I1, I2, I3>(a.0, b.0))
        }
        /// Lanes where `self > 0` keep `v`, others become 0.
        #[inline(always)]
        pub fn keep_where_positive(self, v: F4) -> F4 {
            F4(v128_and(v.0, f32x4_gt(self.0, f32x4_splat(0.0))))
        }
        /// Splits x > 0 into (mantissa in [sqrt(0.5), sqrt(2)) - 1, exponent).
        #[inline(always)]
        pub fn frexp_log(self) -> (F4, F4) {
            let bits = self.0;
            let e = i32x4_sub(u32x4_shr(bits, 23), i32x4_splat(126));
            let m = v128_or(v128_and(bits, u32x4_splat(0x007f_ffff)), u32x4_splat(0x3f00_0000)); // [0.5, 1)
            let small = f32x4_lt(m, f32x4_splat(core::f32::consts::FRAC_1_SQRT_2));
            let e = f32x4_convert_i32x4(i32x4_add(e, small)); // mask is -1 where small
            let m = f32x4_sub(f32x4_add(m, v128_and(m, small)), f32x4_splat(1.0));
            (F4(m), F4(e))
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
mod simd {
    #[derive(Clone, Copy)]
    pub struct F4(pub [f32; 4]);

    macro_rules! lanes {
        ($a:expr, $b:expr, $f:expr) => {{
            let (a, b) = ($a.0, $b.0);
            F4([$f(a[0], b[0]), $f(a[1], b[1]), $f(a[2], b[2]), $f(a[3], b[3])])
        }};
    }

    impl F4 {
        #[inline(always)]
        pub fn splat(x: f32) -> F4 { F4([x; 4]) }
        #[inline(always)]
        pub unsafe fn load(p: *const f32) -> F4 { F4(core::ptr::read_unaligned(p as *const [f32; 4])) }
        #[inline(always)]
        pub unsafe fn store(self, p: *mut f32) { core::ptr::write_unaligned(p as *mut [f32; 4], self.0) }
        #[inline(always)]
        pub fn add(self, o: F4) -> F4 { lanes!(self, o, |a: f32, b: f32| a + b) }
        #[inline(always)]
        pub fn sub(self, o: F4) -> F4 { lanes!(self, o, |a: f32, b: f32| a - b) }
        #[inline(always)]
        pub fn mul(self, o: F4) -> F4 { lanes!(self, o, |a: f32, b: f32| a * b) }
        #[inline(always)]
        pub fn div(self, o: F4) -> F4 { lanes!(self, o, |a: f32, b: f32| a / b) }
        #[inline(always)]
        pub fn min(self, o: F4) -> F4 { lanes!(self, o, |a: f32, b: f32| if b < a { b } else { a }) }
        #[inline(always)]
        pub fn max(self, o: F4) -> F4 { lanes!(self, o, |a: f32, b: f32| if a < b { b } else { a }) }
        #[inline(always)]
        pub fn abs(self) -> F4 { F4(self.0.map(f32::abs)) }
        #[inline(always)]
        pub fn nearest(self) -> F4 { F4(self.0.map(f32::round_ties_even)) }
        #[inline(always)]
        pub fn pow2i(self) -> F4 { F4(self.0.map(|n| f32::from_bits(((n as i32 + 127) as u32) << 23))) }
        #[inline(always)]
        pub fn with_sign_of(self, s: F4) -> F4 { lanes!(self, s, |a: f32, b: f32| a.copysign(b)) }
        #[inline(always)]
        pub fn lane0(self) -> f32 { self.0[0] }
        #[inline(always)]
        pub fn hsum(self) -> f32 { (self.0[0] + self.0[1]) + (self.0[2] + self.0[3]) }
        #[inline(always)]
        pub fn hmax(self) -> f32 { self.0[0].max(self.0[1]).max(self.0[2].max(self.0[3])) }
        #[inline(always)]
        pub fn madd(self, a: F4, b: F4) -> F4 { self.add(a.mul(b)) }
        #[inline(always)]
        pub fn shuf<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(a: F4, b: F4) -> F4 {
            let ab = [a.0[0], a.0[1], a.0[2], a.0[3], b.0[0], b.0[1], b.0[2], b.0[3]];
            F4([ab[I0], ab[I1], ab[I2], ab[I3]])
        }
        #[inline(always)]
        pub fn keep_where_positive(self, v: F4) -> F4 { lanes!(self, v, |x: f32, v: f32| if x > 0.0 { v } else { 0.0 }) }
        #[inline(always)]
        pub fn frexp_log(self) -> (F4, F4) {
            let mut m = [0.0f32; 4];
            let mut e = [0.0f32; 4];
            for i in 0..4 {
                let bits = self.0[i].to_bits();
                let mut ee = (bits >> 23) as i32 - 126;
                let mut mm = f32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000);
                if mm < core::f32::consts::FRAC_1_SQRT_2 {
                    ee -= 1;
                    mm += mm;
                }
                m[i] = mm - 1.0;
                e[i] = ee as f32;
            }
            (F4(m), F4(e))
        }
    }
}

pub use simd::F4;

/// Horner step: p*r + c
#[inline(always)]
fn hs(p: F4, r: F4, c: f32) -> F4 {
    F4::splat(c).madd(p, r)
}

/// Cephes-style expf (~1-2 ulp). Input is clamped to [-87.3, 88], so the 2^n scale stays a normal float.
#[inline(always)]
pub fn exp4(x: F4) -> F4 {
    let x = x.max(F4::splat(-87.33654)).min(F4::splat(88.0));
    let n = x.mul(F4::splat(core::f32::consts::LOG2_E)).nearest();
    let r = x.sub(n.mul(F4::splat(0.693_359_375))).sub(n.mul(F4::splat(-2.121_944_4e-4)));
    let mut p = F4::splat(1.987_569_1e-4);
    p = hs(p, r, 1.398_199_9e-3);
    p = hs(p, r, 8.333_452e-3);
    p = hs(p, r, 4.166_579_6e-2);
    p = hs(p, r, 1.666_666_5e-1);
    p = hs(p, r, 5.000_000_1e-1);
    let y = r.madd(p, r.mul(r)).add(F4::splat(1.0));
    y.mul(n.pow2i())
}

/// Cephes-style logf for positive finite inputs (zero/denormals are clamped to the smallest normal).
#[inline(always)]
pub fn log4(x: F4) -> F4 {
    let (m, e) = x.max(F4::splat(f32::MIN_POSITIVE)).frexp_log();
    let z = m.mul(m);
    let mut p = F4::splat(7.037_683_6e-2);
    p = hs(p, m, -1.151_461e-1);
    p = hs(p, m, 1.167_699_9e-1);
    p = hs(p, m, -1.242_014_1e-1);
    p = hs(p, m, 1.424_932_3e-1);
    p = hs(p, m, -1.666_805_8e-1);
    p = hs(p, m, 2.000_071_4e-1);
    p = hs(p, m, -2.499_999_4e-1);
    p = hs(p, m, 3.333_333e-1);
    let mut y = p.mul(m).mul(z);
    y = y.madd(e, F4::splat(-2.121_944_4e-4));
    y = y.madd(z, F4::splat(-0.5));
    m.add(y).madd(e, F4::splat(0.693_359_375))
}

/// x^e for x >= 0 (x == 0 -> 0 for e > 0).
#[inline(always)]
pub fn pow4(x: F4, e: f32) -> F4 {
    x.keep_where_positive(exp4(log4(x).mul(F4::splat(e))))
}

// -------------------------------------------------------------------------------------------------
// Activations
// -------------------------------------------------------------------------------------------------

pub trait Act {
    fn v(x: F4) -> F4;
    #[inline(always)]
    fn s(x: f32) -> f32 { Self::v(F4::splat(x)).lane0() }
}

pub struct Ident;
pub struct Sigmoid;
pub struct Silu;
pub struct Tanh;
pub struct Exp;

impl Act for Ident {
    #[inline(always)]
    fn v(x: F4) -> F4 { x }
    #[inline(always)]
    fn s(x: f32) -> f32 { x }
}

impl Act for Sigmoid {
    #[inline(always)]
    fn v(x: F4) -> F4 {
        let one = F4::splat(1.0);
        one.div(one.add(exp4(F4::splat(0.0).sub(x))))
    }
}

impl Act for Silu {
    #[inline(always)]
    fn v(x: F4) -> F4 { x.mul(Sigmoid::v(x)) }
}

impl Act for Tanh {
    #[inline(always)]
    fn v(x: F4) -> F4 {
        let one = F4::splat(1.0);
        let e = exp4(F4::splat(-2.0).mul(x.abs()));
        one.sub(e).div(one.add(e)).with_sign_of(x)
    }
}

impl Act for Exp {
    #[inline(always)]
    fn v(x: F4) -> F4 { exp4(x) }
}

#[inline(always)]
pub fn expf(x: f32) -> f32 { Exp::s(x) }

// -------------------------------------------------------------------------------------------------
// Strided n-d iteration helpers
// -------------------------------------------------------------------------------------------------

/// Calls `f` once per innermost row, passing each operand's element offset of the row start.
#[inline(always)]
unsafe fn rows<const K: usize, F: FnMut(&[isize; K])>(shp: &[usize], st: [&[isize]; K], mut f: F) {
    let nd = shp.len();
    debug_assert!(nd >= 1 && nd <= 9);
    let mut idx = [0usize; 8];
    let mut off = [0isize; K];
    loop {
        f(&off);
        let mut d = nd - 1;
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            idx[d] += 1;
            for k in 0..K {
                off[k] += st[k][d];
            }
            if idx[d] < shp[d] {
                break;
            }
            for k in 0..K {
                off[k] -= st[k][d] * shp[d] as isize;
            }
            idx[d] = 0;
        }
    }
}

pub unsafe fn fill(d: *mut f32, n: usize, v: f32) {
    for i in 0..n {
        *d.add(i) = v;
    }
}

pub unsafe fn copy_nd(dst: *mut f32, ds: &[isize], src: *const f32, ss: &[isize], shp: &[usize]) {
    let l = shp.len() - 1;
    let (n, dl, sl) = (shp[l], ds[l], ss[l]);
    rows(shp, [ds, ss], |o| {
        let d = dst.offset(o[0]);
        let s = src.offset(o[1]);
        if dl == 1 && sl == 1 {
            ptr::copy(s, d, n);
        } else if dl == 1 && sl == 0 {
            fill(d, n, *s);
        } else {
            for i in 0..n as isize {
                *d.offset(i * dl) = *s.offset(i * sl);
            }
        }
    });
}

/// out (contiguous) = f(x)
pub unsafe fn map_nd<F: Fn(f32) -> f32>(out: *mut f32, x: *const f32, xs: &[isize], shp: &[usize], f: F) {
    let l = shp.len() - 1;
    let (n, xl) = (shp[l], xs[l]);
    let mut o = out;
    rows(shp, [xs], |off| {
        let s = x.offset(off[0]);
        if xl == 1 {
            for i in 0..n {
                *o.add(i) = f(*s.add(i));
            }
        } else {
            for i in 0..n {
                *o.add(i) = f(*s.offset(i as isize * xl));
            }
        }
        o = o.add(n);
    });
}

/// out (contiguous) = f(a, b)
pub unsafe fn zip_nd<F: Fn(f32, f32) -> f32>(out: *mut f32, a: *const f32, as_: &[isize], b: *const f32,
                                             bs: &[isize], shp: &[usize], f: F) {
    let l = shp.len() - 1;
    let (n, al, bl) = (shp[l], as_[l], bs[l]);
    let mut o = out;
    rows(shp, [as_, bs], |off| {
        let pa = a.offset(off[0]);
        let pb = b.offset(off[1]);
        if al == 1 && bl == 1 {
            for i in 0..n {
                *o.add(i) = f(*pa.add(i), *pb.add(i));
            }
        } else if al == 1 && bl == 0 {
            let vb = *pb;
            for i in 0..n {
                *o.add(i) = f(*pa.add(i), vb);
            }
        } else if al == 0 && bl == 1 {
            let va = *pa;
            for i in 0..n {
                *o.add(i) = f(va, *pb.add(i));
            }
        } else {
            for i in 0..n as isize {
                *o.offset(i) = f(*pa.offset(i * al), *pb.offset(i * bl));
            }
        }
        o = o.add(n);
    });
}

#[inline(always)]
unsafe fn act_contig<A: Act>(out: *mut f32, x: *const f32, n: usize) {
    let mut i = 0;
    while i + 4 <= n {
        A::v(F4::load(x.add(i))).store(out.add(i));
        i += 4;
    }
    while i < n {
        *out.add(i) = A::s(*x.add(i));
        i += 1;
    }
}

pub unsafe fn act_nd<A: Act>(out: *mut f32, x: *const f32, xs: &[isize], shp: &[usize]) {
    let l = shp.len() - 1;
    let (n, xl) = (shp[l], xs[l]);
    let mut o = out;
    rows(shp, [xs], |off| {
        let s = x.offset(off[0]);
        if xl == 1 {
            act_contig::<A>(o, s, n);
        } else {
            for i in 0..n {
                *o.add(i) = A::s(*s.offset(i as isize * xl));
            }
        }
        o = o.add(n);
    });
}

pub unsafe fn act_inplace<A: Act>(p: *mut f32, n: usize) {
    act_contig::<A>(p, p, n);
}

// -------------------------------------------------------------------------------------------------
// Reductions
// -------------------------------------------------------------------------------------------------

pub enum Reduce {
    L1,
    L2,
    Sum,
    SumSquare,
    Mean,
    Max,
    Min,
}

/// x: [outer, red, inner] contiguous -> out: [outer, inner]
pub unsafe fn reduce(out: *mut f32, x: *const f32, outer: usize, red: usize, inner: usize, mode: Reduce) {
    for o in 0..outer {
        for j in 0..inner {
            let base = x.add(o * red * inner + j);
            let mut acc = match mode {
                Reduce::Max => f32::NEG_INFINITY,
                Reduce::Min => f32::INFINITY,
                _ => 0.0,
            };
            for r in 0..red {
                let v = *base.add(r * inner);
                acc = match mode {
                    Reduce::L1 => acc + v.abs(),
                    Reduce::L2 | Reduce::SumSquare => acc + v * v,
                    Reduce::Sum | Reduce::Mean => acc + v,
                    Reduce::Max => acc.max(v),
                    Reduce::Min => acc.min(v),
                };
            }
            *out.add(o * inner + j) = match mode {
                Reduce::L2 => acc.sqrt(),
                Reduce::Mean => acc / red as f32,
                _ => acc,
            };
        }
    }
}

/// Softmax over the middle axis of a contiguous [outer, n, inner] tensor.
pub unsafe fn softmax(out: *mut f32, x: *const f32, outer: usize, n: usize, inner: usize) {
    for o in 0..outer {
        if inner == 1 {
            let xb = x.add(o * n);
            let ob = out.add(o * n);
            let n4 = n & !3;
            let mut m = f32::NEG_INFINITY;
            if n4 > 0 {
                let mut mv = F4::load(xb);
                let mut i = 4;
                while i < n4 {
                    mv = mv.max(F4::load(xb.add(i)));
                    i += 4;
                }
                m = mv.hmax();
            }
            for i in n4..n {
                m = m.max(*xb.add(i));
            }
            let mv = F4::splat(m);
            let mut sv = F4::splat(0.0);
            let mut i = 0;
            while i < n4 {
                let e = exp4(F4::load(xb.add(i)).sub(mv));
                e.store(ob.add(i));
                sv = sv.add(e);
                i += 4;
            }
            let mut s = sv.hsum();
            for i in n4..n {
                let e = expf(*xb.add(i) - m);
                *ob.add(i) = e;
                s += e;
            }
            let inv = F4::splat(1.0 / s);
            let mut i = 0;
            while i < n4 {
                F4::load(ob.add(i)).mul(inv).store(ob.add(i));
                i += 4;
            }
            for i in n4..n {
                *ob.add(i) *= 1.0 / s;
            }
        } else {
            for j in 0..inner {
                let xb = x.add(o * n * inner + j);
                let ob = out.add(o * n * inner + j);
                let mut m = f32::NEG_INFINITY;
                for i in 0..n {
                    m = m.max(*xb.add(i * inner));
                }
                let mut s = 0.0f32;
                for i in 0..n {
                    let e = expf(*xb.add(i * inner) - m);
                    *ob.add(i * inner) = e;
                    s += e;
                }
                let inv = 1.0 / s;
                for i in 0..n {
                    *ob.add(i * inner) *= inv;
                }
            }
        }
    }
}

/// out (contiguous) = x^e  (x >= 0)
pub unsafe fn pow_nd(out: *mut f32, x: *const f32, xs: &[isize], shp: &[usize], e: f32) {
    let l = shp.len() - 1;
    let (n, xl) = (shp[l], xs[l]);
    let mut o = out;
    rows(shp, [xs], |off| {
        let s = x.offset(off[0]);
        let mut i = 0;
        if xl == 1 {
            while i + 4 <= n {
                pow4(F4::load(s.add(i)), e).store(o.add(i));
                i += 4;
            }
        }
        while i < n {
            *o.add(i) = pow4(F4::splat(*s.offset(i as isize * xl)), e).lane0();
            i += 1;
        }
        o = o.add(n);
    });
}

// -------------------------------------------------------------------------------------------------
// GEMM:  C[m x n] = act(alpha * A[m x k] * B[k x n] (+ C) (+ row bias) (+ column bias))
//   A: arbitrary element strides (sam, sak);  B: rows `ldb` apart, unit stride along n;
//   C: row-major, rows `ldc` apart. The epilogue runs on the register tile (no extra memory pass).
// -------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Epi {
    pub alpha: f32,
    pub acc: bool,
    /// bias[m] added to every row (or null)
    pub brow: *const f32,
    /// bias[n] added to every column (or null)
    pub bcol: *const f32,
}

impl Epi {
    pub const NONE: Epi = Epi { alpha: 1.0, acc: false, brow: ptr::null(), bcol: ptr::null() };
    #[inline(always)]
    pub const fn rows(b: *const f32) -> Epi { Epi { brow: b, ..Epi::NONE } }
    #[inline(always)]
    pub const fn cols(b: *const f32) -> Epi { Epi { bcol: b, ..Epi::NONE } }
    #[inline(always)]
    pub const fn acc() -> Epi { Epi { acc: true, ..Epi::NONE } }
    #[inline(always)]
    pub const fn alpha(self, alpha: f32) -> Epi { Epi { alpha, ..self } }
}

#[inline(always)]
unsafe fn tile<A: Act, const MR: usize, const NV: usize>(k: usize, a: *const f32, sam: isize, sak: isize,
                                                         b: *const f32, ldb: usize, c: *mut f32, ldc: usize,
                                                         epi: &Epi, i0: usize, j0: usize) {
    let mut cc = [[F4::splat(0.0); NV]; MR];
    let mut pa = a;
    let mut pb = b;
    for _ in 0..k {
        let mut bv = [F4::splat(0.0); NV];
        for v in 0..NV {
            bv[v] = F4::load(pb.add(4 * v));
        }
        for r in 0..MR {
            let av = F4::splat(*pa.offset(r as isize * sam));
            for v in 0..NV {
                cc[r][v] = cc[r][v].madd(av, bv[v]);
            }
        }
        pa = pa.offset(sak);
        pb = pb.add(ldb);
    }
    let al = F4::splat(epi.alpha);
    let mut bc = [F4::splat(0.0); NV];
    if !epi.bcol.is_null() {
        for v in 0..NV {
            bc[v] = F4::load(epi.bcol.add(j0 + 4 * v));
        }
    }
    for r in 0..MR {
        let pc = c.add(r * ldc);
        let br = if epi.brow.is_null() { None } else { Some(F4::splat(*epi.brow.add(i0 + r))) };
        for v in 0..NV {
            let mut val = if epi.alpha == 1.0 { cc[r][v] } else { cc[r][v].mul(al) };
            if epi.acc {
                val = val.add(F4::load(pc.add(4 * v)));
            }
            if let Some(br) = br {
                val = val.add(br);
            }
            if !epi.bcol.is_null() {
                val = val.add(bc[v]);
            }
            A::v(val).store(pc.add(4 * v));
        }
    }
}

#[inline(always)]
unsafe fn tile_scalar<A: Act, const MR: usize>(k: usize, a: *const f32, sam: isize, sak: isize, b: *const f32,
                                               ldb: usize, c: *mut f32, ldc: usize, epi: &Epi, i0: usize, j0: usize) {
    let mut cc = [0.0f32; MR];
    let mut pa = a;
    let mut pb = b;
    for _ in 0..k {
        let bv = *pb;
        for r in 0..MR {
            cc[r] += *pa.offset(r as isize * sam) * bv;
        }
        pa = pa.offset(sak);
        pb = pb.add(ldb);
    }
    for r in 0..MR {
        let pc = c.add(r * ldc);
        let mut v = cc[r] * epi.alpha;
        if epi.acc {
            v += *pc;
        }
        if !epi.brow.is_null() {
            v += *epi.brow.add(i0 + r);
        }
        if !epi.bcol.is_null() {
            v += *epi.bcol.add(j0);
        }
        *pc = A::s(v);
    }
}

#[inline(always)]
unsafe fn row_block<A: Act, const MR: usize>(n: usize, k: usize, a: *const f32, sam: isize, sak: isize,
                                             b: *const f32, ldb: usize, c: *mut f32, ldc: usize, epi: &Epi, i0: usize) {
    // V8 on x64 only has 16 XMM registers: keep MR*NV accumulators + NV + 1 operands within that.
    let mut j = 0;
    while j + 8 <= n {
        tile::<A, MR, 2>(k, a, sam, sak, b.add(j), ldb, c.add(j), ldc, epi, i0, j);
        j += 8;
    }
    while j + 4 <= n {
        tile::<A, MR, 1>(k, a, sam, sak, b.add(j), ldb, c.add(j), ldc, epi, i0, j);
        j += 4;
    }
    while j < n {
        tile_scalar::<A, MR>(k, a, sam, sak, b.add(j), ldb, c.add(j), ldc, epi, i0, j);
        j += 1;
    }
}

pub unsafe fn sgemm_ex<A: Act>(m: usize, n: usize, k: usize, a: *const f32, sam: isize, sak: isize,
                               b: *const f32, ldb: usize, c: *mut f32, ldc: usize, epi: Epi) {
    let mut i = 0;
    while i + 4 <= m {
        row_block::<A, 4>(n, k, a.offset(i as isize * sam), sam, sak, b, ldb, c.add(i * ldc), ldc, &epi, i);
        i += 4;
    }
    while i + 2 <= m {
        row_block::<A, 2>(n, k, a.offset(i as isize * sam), sam, sak, b, ldb, c.add(i * ldc), ldc, &epi, i);
        i += 2;
    }
    while i < m {
        row_block::<A, 1>(n, k, a.offset(i as isize * sam), sam, sak, b, ldb, c.add(i * ldc), ldc, &epi, i);
        i += 1;
    }
}

pub unsafe fn sgemm(m: usize, n: usize, k: usize, alpha: f32, a: *const f32, sam: isize, sak: isize,
                    b: *const f32, ldb: usize, c: *mut f32, ldc: usize, acc: bool) {
    let epi = if acc { Epi::acc() } else { Epi::NONE };
    sgemm_ex::<Ident>(m, n, k, a, sam, sak, b, ldb, c, ldc, epi.alpha(alpha));
}

/// Batched GEMM; `out` is contiguous [batch..., m, n].
pub unsafe fn matmul_batched(out: *mut f32, a: *const f32, as_: &[isize], sam: isize, sak: isize, b: *const f32,
                             bs: &[isize], ldb: usize, shp: &[usize], m: usize, n: usize, k: usize) {
    let l = shp.len() - 1;
    let (nl, al, bl) = (shp[l], as_[l], bs[l]);
    let mut o = out;
    rows(shp, [as_, bs], |off| {
        for i in 0..nl as isize {
            sgemm(m, n, k, 1.0, a.offset(off[0] + i * al), sam, sak, b.offset(off[1] + i * bl), ldb, o, n, false);
            o = o.add(m * n);
        }
    });
}

/// C[r, :] = bias[r]
pub unsafe fn bias_rows(c: *mut f32, bias: *const f32, rows_: usize, cols: usize) {
    for r in 0..rows_ {
        fill(c.add(r * cols), cols, *bias.add(r));
    }
}

// -------------------------------------------------------------------------------------------------
// 1-D convolution helpers
// -------------------------------------------------------------------------------------------------

/// col[(c*K + kk), t] = x[c, t*s + kk*d - pl]   (zero outside [0, L))
pub unsafe fn im2col_1d(col: *mut f32, x: *const f32, cin: usize, xs: isize, l: usize, k: usize, s: usize,
                        d: usize, pl: usize, lout: usize) {
    for c in 0..cin {
        let xr = x.offset(c as isize * xs);
        for kk in 0..k {
            let row = col.add((c * k + kk) * lout);
            let shift = (kk * d) as isize - pl as isize;
            if s == 1 {
                // valid t: 0 <= t + shift < l
                let t0 = (-shift).clamp(0, lout as isize) as usize;
                let t1 = (l as isize - shift).clamp(t0 as isize, lout as isize) as usize;
                fill(row, t0, 0.0);
                ptr::copy_nonoverlapping(xr.offset(t0 as isize + shift), row.add(t0), t1 - t0);
                fill(row.add(t1), lout - t1, 0.0);
            } else {
                for t in 0..lout {
                    let pos = (t * s) as isize + shift;
                    *row.add(t) = if pos >= 0 && (pos as usize) < l { *xr.offset(pos) } else { 0.0 };
                }
            }
        }
    }
}

// Winograd F(2,3) for K=3 / stride 1 / pad 1 convolutions (1.5x fewer MACs than im2col + GEMM).
//   V_i = B^T d   per input channel and output pair (d = x[2j-1 .. 2j+2]):
//     V0 = d0 - d2,  V1 = d1 + d2,  V2 = d2 - d1,  V3 = d1 - d3
//   M_i = U_i . V_i  (four GEMMs; U_i = G g are precomputed weights)
//   y[2j] = M0 + M1 + M2,  y[2j+1] = M1 - M2 - M3
// With x split into even/odd samples (E[j] = x[2j], O[j] = x[2j+1]) every term is a contiguous vector:
//   d0 = O[j-1], d1 = E[j], d2 = O[j], d3 = E[j+1].

/// x: [cin, l] (row stride xs, l % 8 == 0) -> v: [4][cin][l/2]
pub unsafe fn wino23_in(v: *mut f32, x: *const f32, cin: usize, xs: isize, l: usize) {
    let h = l / 2;
    let plane = cin * h;
    let z = F4::splat(0.0);
    for c in 0..cin {
        let xr = x.offset(c as isize * xs);
        let vr = v.add(c * h);
        let load_eo = |j: usize| {
            let (a, b) = (F4::load(xr.add(2 * j)), F4::load(xr.add(2 * j + 4)));
            (F4::shuf::<0, 2, 4, 6>(a, b), F4::shuf::<1, 3, 5, 7>(a, b))
        };
        let (mut e, mut o) = load_eo(0);
        let mut prev_o = z;
        let mut j = 0;
        while j < h {
            let (ne, no) = if j + 4 < h { load_eo(j + 4) } else { (z, z) };
            let o_prev = F4::shuf::<3, 4, 5, 6>(prev_o, o);
            let e_next = F4::shuf::<1, 2, 3, 4>(e, ne);
            o_prev.sub(o).store(vr.add(j));
            e.add(o).store(vr.add(plane + j));
            o.sub(e).store(vr.add(2 * plane + j));
            e.sub(e_next).store(vr.add(3 * plane + j));
            prev_o = o;
            e = ne;
            o = no;
            j += 4;
        }
    }
}

/// m: [4][cout][l/2] -> y: [cout, l] = act(output transform + bias[co])
pub unsafe fn wino23_out<A: Act>(y: *mut f32, m: *const f32, cout: usize, l: usize, bias: *const f32) {
    let h = l / 2;
    let plane = cout * h;
    for co in 0..cout {
        let bv = F4::splat(if bias.is_null() { 0.0 } else { *bias.add(co) });
        let mr = m.add(co * h);
        let yr = y.add(co * l);
        let mut j = 0;
        while j < h {
            let m0 = F4::load(mr.add(j));
            let m1 = F4::load(mr.add(plane + j));
            let m2 = F4::load(mr.add(2 * plane + j));
            let m3 = F4::load(mr.add(3 * plane + j));
            let y0 = A::v(m0.add(m1).add(m2).add(bv));
            let y1 = A::v(m1.sub(m2).sub(m3).add(bv));
            F4::shuf::<0, 4, 1, 5>(y0, y1).store(yr.add(2 * j));
            F4::shuf::<2, 6, 3, 7>(y0, y1).store(yr.add(2 * j + 4));
            j += 4;
        }
    }
}

/// out[co, l*s + kk*d - pl] += col[(co*K + kk), l]
pub unsafe fn col2im_1d(out: *mut f32, col: *const f32, cout: usize, k: usize, l: usize, s: usize, d: usize,
                        pl: usize, lout: usize) {
    for co in 0..cout {
        let orow = out.add(co * lout);
        for kk in 0..k {
            let crow = col.add((co * k + kk) * l);
            let shift = (kk * d) as isize - pl as isize;
            for t in 0..l {
                let pos = (t * s) as isize + shift;
                if pos >= 0 && (pos as usize) < lout {
                    *orow.offset(pos) += *crow.add(t);
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// DFT (ONNX semantics: [.., n, c] with c = 1 real / 2 complex; inverse is scaled by 1/n)
// -------------------------------------------------------------------------------------------------

/// `tw`: n complex twiddles exp(-2*pi*i*k/n); `scratch`: >= 2n floats.
pub unsafe fn dft(out: *mut f32, x: *const f32, outer: usize, n: usize, c: usize, inverse: bool, onesided: bool,
                  tw: *const f32, scratch: *mut f32) {
    let nout = if onesided { n / 2 + 1 } else { n };
    let sign = if inverse { -1.0f32 } else { 1.0 };
    let scale = if inverse { 1.0 / n as f32 } else { 1.0 };
    let buf = scratch;
    for o in 0..outer {
        let xin = x.add(o * n * c);
        let dst = out.add(o * nout * 2);
        if n.is_power_of_two() {
            let bits = n.trailing_zeros();
            for i in 0..n {
                let j = if n == 1 { 0 } else { i.reverse_bits() >> (usize::BITS - bits) };
                *buf.add(2 * j) = *xin.add(i * c);
                *buf.add(2 * j + 1) = if c == 2 { *xin.add(i * c + 1) } else { 0.0 };
            }
            let mut len = 2;
            while len <= n {
                let half = len / 2;
                let step = n / len;
                let mut start = 0;
                while start < n {
                    for j in 0..half {
                        let wr = *tw.add(2 * j * step);
                        let wi = *tw.add(2 * j * step + 1) * sign;
                        let pa = buf.add(2 * (start + j));
                        let pb = buf.add(2 * (start + j + half));
                        let (br, bi) = (*pb, *pb.add(1));
                        let tr = wr * br - wi * bi;
                        let ti = wr * bi + wi * br;
                        let (ar, ai) = (*pa, *pa.add(1));
                        *pb = ar - tr;
                        *pb.add(1) = ai - ti;
                        *pa = ar + tr;
                        *pa.add(1) = ai + ti;
                    }
                    start += len;
                }
                len <<= 1;
            }
            for kk in 0..nout {
                *dst.add(2 * kk) = *buf.add(2 * kk) * scale;
                *dst.add(2 * kk + 1) = *buf.add(2 * kk + 1) * scale;
            }
        } else {
            for kk in 0..nout {
                let (mut sr, mut si) = (0.0f32, 0.0f32);
                for j in 0..n {
                    let t = (j * kk) % n;
                    let wr = *tw.add(2 * t);
                    let wi = *tw.add(2 * t + 1) * sign;
                    let xr = *xin.add(j * c);
                    let xi = if c == 2 { *xin.add(j * c + 1) } else { 0.0 };
                    sr += xr * wr - xi * wi;
                    si += xr * wi + xi * wr;
                }
                *buf.add(2 * kk) = sr * scale;
                *buf.add(2 * kk + 1) = si * scale;
            }
            ptr::copy_nonoverlapping(buf, dst, 2 * nout);
        }
    }
}
