//! Standalone reproduction of the Metal-only zensim-gpu diffmap divergence
//! (imazen/zenmetrics#20). Depends on **cubecl only** — no zensim or sibling
//! repos — so it can be handed to a Metal-equipped session or to the
//! gfx-rs/wgpu (naga) maintainers as a self-contained artifact.
//!
//! ## What it does
//!
//! It drives the exact zensim-gpu diffmap kernel chain in isolation:
//!
//!   for each pyramid scale s:
//!     1. `diffmap_zero_kernel`            — zero scale_dm[s]   (toggle, see below)
//!     2. `per_scale_weighted_ssim_kernel` — synthetic mu/ssq/s12 planes -> scale_dm[s]
//!     3. `pow2x_upsample_add_kernel`      — NN-replicate scale_dm[s] (x2^s) into acc
//!
//! then reads `acc` back and compares it, pixel-by-pixel, against a plain-Rust
//! CPU computation of the identical math. The input planes are deterministic
//! synthetic data (no image decode, no feature extraction) — only the **buffer
//! geometry** (multi-scale, channel-concatenated `[ch0|ch1|ch2]` layout, NN
//! upsample) matches the real pipeline, which is what an indexing / stale-read
//! codegen bug keys off of.
//!
//! ## Expected results (measured upstream of this repro, in the full pipeline)
//!
//! - CUDA  (`--features cuda`)  : matches the CPU reference (~1e-4). PASS.
//! - Vulkan (`--features wgpu` on Linux/Windows-NVIDIA) : matches. PASS.
//! - Metal  (`--features wgpu` on macOS) : a scattered subset of `acc` pixels at
//!   sizes >= 96 come back holding a fixed value INDEPENDENT of the inputs
//!   (~1.098 in the real metric). FAIL.
//!
//! ## Run
//!
//! ```bash
//! # macOS (wgpu picks Metal) — expected to FAIL at 96x80:
//! cargo run --release --no-default-features --features wgpu
//! # Linux/NVIDIA (wgpu picks Vulkan) — expected to PASS (proves the WGSL is fine):
//! cargo run --release --no-default-features --features wgpu
//! # CUDA — expected to PASS:
//! cargo run --release --no-default-features --features cuda
//! ```
//!
//! `ZERO_FILL=0` env var disables step 1 (the `648a8c7b` mitigation) so you can
//! A/B whether zeroing scale_dm collapses the Metal divergence. `ZERO_FILL=1`
//! (default) enables it.

// The CPU reference loops index by position deliberately (mirroring the GPU
// kernels' flat indexing) — clearest read for a reproduction.
#![allow(clippy::needless_range_loop)]

use cubecl::prelude::*;

// ───────────── feature-kernel constants (verbatim from zensim-gpu fused.rs) ─────────────
// TX=64 (2 warps/block); R=5 / DIAM=11 is the 11-tap separable blur radius/diameter.
const TX: u32 = 64;
const R: u32 = 5;
const DIAM: u32 = 11;
const TILE_COLS: u32 = TX + 2u32 * R;
const TILE_COLS_US: usize = (TX + 2u32 * R) as usize;
const BUF_LEN_US: usize = (DIAM * TX) as usize;
const C2: f32 = 0.0009;
const INV_DIAM: f32 = 1.0 / 11.0;

// ───────────── df64 (double-single) precision probe — Thall 2006 / Dekker ─────────────
// Extended precision as an unevaluated sum of two f32 (~44 mantissa bits) for GPUs
// without native f64 (Apple/Metal). Pure f32 + the Knuth two-sum error-free
// transform — which stays exact ONLY if the compiler does not reassociate. Metal
// defaults to fast-math, which can collapse the compensation terms to zero
// (degrading df64 back to f32). This probe tests exactly that on real Metal CI
// before we commit to a df64 `CubeType` in cubecl-std. Inlined as f32 hi/lo locals
// (the struct form needs CubeType assignment plumbing we add in cubecl-std later).
//
// base=2^24, add 1.0 x n: each add is below the f32 ulp at 2^24 (=2), so naive f32
// loses it while df64 recovers ~n. Args are runtime so the loop can't be folded.
// out[0] = f32 recovered (≈0), out[1] = df64 recovered (≈n if the transform survives).
//
// `precise` marks this kernel precision-critical: on Metal the per-kernel control
// disables MTLCompileOptions fast-math for THIS module only, so the df64 two-sum
// survives (recovers ~4096). Other (non-precise) kernels keep fast-math on.
#[cube(launch_unchecked, precise)]
fn df64_probe_kernel(out: &mut Array<f32>, base: f32, add_val: f32, n: u32) {
    let mut sf = base;
    let mut i = 0u32;
    while i < n {
        sf += add_val;
        i += 1u32;
    }
    out[0] = sf - base;

    let mut hi = base;
    let mut lo = f32::new(0.0);
    let mut j = 0u32;
    while j < n {
        // Knuth two-sum of (hi, add_val), carrying lo, then renormalize.
        let s = hi + add_val;
        let bb = s - hi;
        let e = (hi - (s - bb)) + (add_val - bb);
        let nlo = e + lo;
        hi = s + nlo;
        lo = nlo - (hi - s);
        j += 1u32;
    }
    out[1] = (hi - base) + lo;
}

/// Identical to `df64_probe_kernel` but WITHOUT `precise` — keeps Metal's default
/// fast-math, so its df64 two-sum collapses (recovers ~0). Side-by-side with the
/// precise twin in one run, this proves the per-kernel control: same machine, same
/// backend, only the `precise` attribute differs.
#[cube(launch_unchecked)]
fn df64_probe_fast_kernel(out: &mut Array<f32>, base: f32, add_val: f32, n: u32) {
    let mut sf = base;
    let mut i = 0u32;
    while i < n {
        sf += add_val;
        i += 1u32;
    }
    out[0] = sf - base;

    let mut hi = base;
    let mut lo = f32::new(0.0);
    let mut j = 0u32;
    while j < n {
        let s = hi + add_val;
        let bb = s - hi;
        let e = (hi - (s - bb)) + (add_val - bb);
        let nlo = e + lo;
        hi = s + nlo;
        lo = nlo - (hi - s);
        j += 1u32;
    }
    out[1] = (hi - base) + lo;
}

// ───────────────────────── kernels (verbatim from ─────────────────────────
// zensim-gpu/src/kernels/diffmap.rs — keep byte-identical so the repro
// exercises the same code the metric ships).

#[cube(launch_unchecked)]
fn diffmap_zero_kernel(dest: &mut Array<f32>, n: u32) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    dest[idx] = f32::new(0.0);
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn per_scale_weighted_ssim_kernel(
    mu1_all: &Array<f32>,
    mu2_all: &Array<f32>,
    ssq_all: &Array<f32>,
    s12_all: &Array<f32>,
    out: &mut Array<f32>,
    padded_w: u32,
    height: u32,
    pad_total: u32,
    w_x: f32,
    w_y: f32,
    w_b: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (padded_w * height) as usize;
    if idx >= total {
        terminate!();
    }
    let pt = pad_total as usize;
    let c2: f32 = f32::new(0.0009);
    let one: f32 = f32::new(1.0);
    let two: f32 = f32::new(2.0);
    let zero: f32 = f32::new(0.0);

    let m1_x = mu1_all[idx];
    let m2_x = mu2_all[idx];
    let sq_x = ssq_all[idx];
    let s12_x = s12_all[idx];
    let mu_diff_x = m1_x - m2_x;
    let num_m_x = fma(mu_diff_x, -mu_diff_x, one);
    let inner_ns_x = fma(-m1_x, m2_x, s12_x);
    let num_s_x = fma(two, inner_ns_x, c2);
    let inner_ds_x = fma(-m1_x, m1_x, sq_x);
    let denom_s_x = fma(-m2_x, m2_x, inner_ds_x) + c2;
    let sd_raw_x = one - (num_m_x * num_s_x) / denom_s_x;
    let sd_x = if sd_raw_x > zero { sd_raw_x } else { zero };

    let m1_y = mu1_all[idx + pt];
    let m2_y = mu2_all[idx + pt];
    let sq_y = ssq_all[idx + pt];
    let s12_y = s12_all[idx + pt];
    let mu_diff_y = m1_y - m2_y;
    let num_m_y = fma(mu_diff_y, -mu_diff_y, one);
    let inner_ns_y = fma(-m1_y, m2_y, s12_y);
    let num_s_y = fma(two, inner_ns_y, c2);
    let inner_ds_y = fma(-m1_y, m1_y, sq_y);
    let denom_s_y = fma(-m2_y, m2_y, inner_ds_y) + c2;
    let sd_raw_y = one - (num_m_y * num_s_y) / denom_s_y;
    let sd_y = if sd_raw_y > zero { sd_raw_y } else { zero };

    let m1_b = mu1_all[idx + pt * 2];
    let m2_b = mu2_all[idx + pt * 2];
    let sq_b = ssq_all[idx + pt * 2];
    let s12_b = s12_all[idx + pt * 2];
    let mu_diff_b = m1_b - m2_b;
    let num_m_b = fma(mu_diff_b, -mu_diff_b, one);
    let inner_ns_b = fma(-m1_b, m2_b, s12_b);
    let num_s_b = fma(two, inner_ns_b, c2);
    let inner_ds_b = fma(-m1_b, m1_b, sq_b);
    let denom_s_b = fma(-m2_b, m2_b, inner_ds_b) + c2;
    let sd_raw_b = one - (num_m_b * num_s_b) / denom_s_b;
    let sd_b = if sd_raw_b > zero { sd_raw_b } else { zero };

    out[idx] = w_x * sd_x + w_y * sd_y + w_b * sd_b;
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pow2x_upsample_add_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    log2_factor: u32,
    blend_weight: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }
    let dw = dst_w as usize;
    let sw = src_w as usize;
    let dx = idx % dw;
    let dy = idx / dw;

    let sx = dx >> log2_factor as usize;
    let sy = dy >> log2_factor as usize;

    let last_sx = src_w as usize - 1usize;
    let last_sy = src_h as usize - 1usize;
    let sx_c = if sx < last_sx { sx } else { last_sx };
    let sy_c = if sy < last_sy { sy } else { last_sy };

    let v = src[sy_c * sw + sx_c];
    dst[idx] = dst[idx] + blend_weight * v;
}

// ───────────────────── GPU producer kernels (the new variable) ─────────────────────
// The isolated host-uploaded chain PASSES on Metal. The real pipeline GPU-PRODUCES
// the mu1/mu2/ssq/s12 planes (via a 3D channel-as-z dispatch) before `per_scale`
// reads them cross-channel (idx, idx+pt, idx+pt*2). These relay kernels reproduce
// "the persist planes are written by a prior dispatch" without porting the whole
// feature kernel: they copy host-staged synth values into the planes on-device.
// The values are byte-identical to the host-upload path, so the CPU reference is
// unchanged — only HOW the planes get filled differs (host blit vs GPU dispatch).

/// 1D relay: flat copy tmp -> plane over the whole concatenated [ch0|ch1|ch2] buffer.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn relay_planes_1d_kernel(
    tmp1: &Array<f32>,
    tmp2: &Array<f32>,
    tmp3: &Array<f32>,
    tmp4: &Array<f32>,
    p1: &mut Array<f32>,
    p2: &mut Array<f32>,
    p3: &mut Array<f32>,
    p4: &mut Array<f32>,
    n: u32, // plane_len = pad_total * 3
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    p1[idx] = tmp1[idx];
    p2[idx] = tmp2[idx];
    p3[idx] = tmp3[idx];
    p4[idx] = tmp4[idx];
}

/// 3D relay: dispatched (cube_x, 1, 3) with channel = CUBE_POS_Z, exactly like
/// `fused_features_kernel_persist`'s grid shape — each channel's plane region
/// [channel*pt, channel*pt + pt) is written by a *different* threadgroup-z, then
/// `per_scale` reads all three regions from one thread. This is the producer→
/// consumer storage pattern the real pipeline exercises and this repro did not.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn relay_planes_3d_kernel(
    tmp1: &Array<f32>,
    tmp2: &Array<f32>,
    tmp3: &Array<f32>,
    tmp4: &Array<f32>,
    p1: &mut Array<f32>,
    p2: &mut Array<f32>,
    p3: &mut Array<f32>,
    p4: &mut Array<f32>,
    pad_total: u32,
) {
    let channel = CUBE_POS_Z;
    let pixel = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    if pixel >= pad_total {
        terminate!();
    }
    let off = (channel as usize) * (pad_total as usize) + pixel as usize;
    p1[off] = tmp1[off];
    p2[off] = tmp2[off];
    p3[off] = tmp3[off];
    p4[off] = tmp4[off];
}

/// Which producer the feature harness launches.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FeatMode {
    /// Full `fused_features_kernel_persist` (f64 partials + SSIM/feature math). FAILS on Metal.
    Full,
    /// Blur-only (no f64, no partials, no feature math). PASSES on Metal.
    BlurOnly,
    /// Blur-only PLUS a single f64 accumulator written to an f64 buffer. The f64 isolator.
    BlurF64,
}

impl FeatMode {
    fn label(self) -> &'static str {
        match self {
            FeatMode::Full => "feature-full",
            FeatMode::BlurOnly => "blur-only   ",
            FeatMode::BlurF64 => "blur+f64    ",
        }
    }
}

// ───── f64 isolator: blur-only + one f64 accumulator + f64 buffer write ─────
// Identical to `blur_planes_kernel` (which PASSES on Metal) plus the MINIMUM f64:
// an `a0: f64` accumulated each row and written to an `Array<f64>`. naga's MSL
// backend advertises no FLOAT64 yet its writer emits `double`; if merely adding
// this f64 makes the f32 plane writes go wrong on Metal, f64 is the trigger.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn blur_f64_kernel(
    src_a: &Array<f32>,
    dst_a: &Array<f32>,
    src_b: &Array<f32>,
    dst_b: &Array<f32>,
    src_c: &Array<f32>,
    dst_c: &Array<f32>,
    partials_f64: &mut Array<f64>,
    mu1_all: &mut Array<f32>,
    mu2_all: &mut Array<f32>,
    ssq_all: &mut Array<f32>,
    s12_all: &mut Array<f32>,
    width: u32,
    height: u32,
    n_strips: u32,
    pad_total: u32,
) {
    let tx = UNIT_POS_X;
    let col_block = CUBE_POS_X;
    let strip = CUBE_POS_Y;
    let channel = CUBE_POS_Z;
    let col_base = col_block * TX;
    let col = col_base + tx;
    let in_bounds = col < width;

    let w = width as usize;
    let pw = width as usize;
    let n_strips_us = n_strips as usize;
    let pt = pad_total as usize;
    let ch_base = (channel as usize) * pt;
    let period_x = 2u32 * (width - 1u32);
    let period_y = 2u32 * (height - 1u32);

    let strip_h_base = height / n_strips;
    let strip_rem = height - strip_h_base * n_strips;
    let y_start = strip * strip_h_base + u32::min(strip, strip_rem);
    let y_end_unclamp = y_start + strip_h_base + (if strip < strip_rem { 1u32 } else { 0u32 });
    let y_end = u32::min(y_end_unclamp, height);

    let mut src_row = SharedMemory::<f32>::new(TILE_COLS_US);
    let mut dst_row = SharedMemory::<f32>::new(TILE_COLS_US);
    let mut buf_mu1 = SharedMemory::<f32>::new(BUF_LEN_US);
    let mut buf_mu2 = SharedMemory::<f32>::new(BUF_LEN_US);
    let mut buf_sq = SharedMemory::<f32>::new(BUF_LEN_US);
    let mut buf_s12 = SharedMemory::<f32>::new(BUF_LEN_US);

    let mut sum_m1 = 0.0_f32;
    let mut sum_m2 = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut sum_s12 = 0.0_f32;
    let mut a0 = 0.0_f64; // the ONLY f64

    let mut k: u32 = 0u32;
    while k < DIAM {
        let raw_y = (y_start + k + period_y - R) % period_y;
        let y_in = if raw_y < height {
            raw_y
        } else {
            period_y - raw_y
        };
        sync_cube();
        let mut i: u32 = 0u32;
        while i * TX + tx < TILE_COLS {
            let load_x = i * TX + tx;
            let raw_x = (col_base + load_x + period_x - R) % period_x;
            let gx = if raw_x < width {
                raw_x
            } else {
                period_x - raw_x
            };
            let off = (y_in as usize) * w + (gx as usize);
            let s_val = if channel == 0u32 {
                src_a[off]
            } else if channel == 1u32 {
                src_b[off]
            } else {
                src_c[off]
            };
            let d_val = if channel == 0u32 {
                dst_a[off]
            } else if channel == 1u32 {
                dst_b[off]
            } else {
                dst_c[off]
            };
            src_row[load_x as usize] = s_val;
            dst_row[load_x as usize] = d_val;
            i += 1u32;
        }
        sync_cube();
        let mut m1 = 0.0_f32;
        let mut m2 = 0.0_f32;
        let mut sq = 0.0_f32;
        let mut s12 = 0.0_f32;
        let mut j: u32 = 0u32;
        while j < DIAM {
            let s = src_row[(tx + j) as usize];
            let d = dst_row[(tx + j) as usize];
            m1 += s;
            m2 += d;
            sq = fma(s, s, fma(d, d, sq));
            s12 = fma(s, d, s12);
            j += 1u32;
        }
        m1 *= INV_DIAM;
        m2 *= INV_DIAM;
        sq *= INV_DIAM;
        s12 *= INV_DIAM;
        let buf_idx = (k * TX + tx) as usize;
        buf_mu1[buf_idx] = m1;
        buf_mu2[buf_idx] = m2;
        buf_sq[buf_idx] = sq;
        buf_s12[buf_idx] = s12;
        sum_m1 += m1;
        sum_m2 += m2;
        sum_sq += sq;
        sum_s12 += s12;
        k += 1u32;
    }

    let mut slot: u32 = 0u32;
    let mut y: u32 = y_start;
    while y < y_end {
        let mu1 = sum_m1 * INV_DIAM;
        let mu2 = sum_m2 * INV_DIAM;
        let ssq = sum_sq * INV_DIAM;
        let s12_v = sum_s12 * INV_DIAM;

        let off = (y as usize) * w + (col as usize);
        if in_bounds {
            mu1_all[ch_base + off] = mu1;
            mu2_all[ch_base + off] = mu2;
            ssq_all[ch_base + off] = ssq;
            s12_all[ch_base + off] = s12_v;
        }
        a0 += mu1 as f64; // the f64 work

        let buf_idx = (slot * TX + tx) as usize;
        let old_m1 = buf_mu1[buf_idx];
        let old_m2 = buf_mu2[buf_idx];
        let old_sq = buf_sq[buf_idx];
        let old_s12 = buf_s12[buf_idx];

        let raw_y = (y + R + 1u32 + period_y) % period_y;
        let y_in = if raw_y < height {
            raw_y
        } else {
            period_y - raw_y
        };
        sync_cube();
        let mut i: u32 = 0u32;
        while i * TX + tx < TILE_COLS {
            let load_x = i * TX + tx;
            let raw_x = (col_base + load_x + period_x - R) % period_x;
            let gx = if raw_x < width {
                raw_x
            } else {
                period_x - raw_x
            };
            let off2 = (y_in as usize) * w + (gx as usize);
            let s_val = if channel == 0u32 {
                src_a[off2]
            } else if channel == 1u32 {
                src_b[off2]
            } else {
                src_c[off2]
            };
            let d_val = if channel == 0u32 {
                dst_a[off2]
            } else if channel == 1u32 {
                dst_b[off2]
            } else {
                dst_c[off2]
            };
            src_row[load_x as usize] = s_val;
            dst_row[load_x as usize] = d_val;
            i += 1u32;
        }
        sync_cube();
        let mut nm1 = 0.0_f32;
        let mut nm2 = 0.0_f32;
        let mut nsq = 0.0_f32;
        let mut ns12 = 0.0_f32;
        let mut j: u32 = 0u32;
        while j < DIAM {
            let s = src_row[(tx + j) as usize];
            let d = dst_row[(tx + j) as usize];
            nm1 += s;
            nm2 += d;
            nsq = fma(s, s, fma(d, d, nsq));
            ns12 = fma(s, d, ns12);
            j += 1u32;
        }
        nm1 *= INV_DIAM;
        nm2 *= INV_DIAM;
        nsq *= INV_DIAM;
        ns12 *= INV_DIAM;
        sum_m1 = sum_m1 + nm1 - old_m1;
        sum_m2 = sum_m2 + nm2 - old_m2;
        sum_sq = sum_sq + nsq - old_sq;
        sum_s12 = sum_s12 + ns12 - old_s12;
        buf_mu1[buf_idx] = nm1;
        buf_mu2[buf_idx] = nm2;
        buf_sq[buf_idx] = nsq;
        buf_s12[buf_idx] = ns12;
        slot = (slot + 1u32) % DIAM;
        y += 1u32;
    }

    if !in_bounds {
        terminate!();
    }
    let slot_idx_us =
        (channel as usize) * n_strips_us * pw + (strip as usize) * pw + (col as usize);
    partials_f64[slot_idx_us] = a0;
}

// ───────────── minimized producer: blur-only (no f64, no partials, no SSIM math) ─────────────
// Same shared-memory cooperative load + sliding-window 11×11 mirror box blur as
// `fused_features_kernel_persist`, but stripped to ONLY write the mu1/mu2/ssq/s12
// planes. Removes: f64 partials buffer, a0..a16/peak accumulators, the SSIM/
// artifact/detail feature math, and the y_body mask. If THIS still miscompiles on
// Metal, the defect is in the core blur (shared memory / sliding window / mirror
// indexing), not the f64 or feature path.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn blur_planes_kernel(
    src_a: &Array<f32>,
    dst_a: &Array<f32>,
    src_b: &Array<f32>,
    dst_b: &Array<f32>,
    src_c: &Array<f32>,
    dst_c: &Array<f32>,
    mu1_all: &mut Array<f32>,
    mu2_all: &mut Array<f32>,
    ssq_all: &mut Array<f32>,
    s12_all: &mut Array<f32>,
    width: u32,
    height: u32,
    n_strips: u32,
    pad_total: u32,
) {
    let tx = UNIT_POS_X;
    let col_block = CUBE_POS_X;
    let strip = CUBE_POS_Y;
    let channel = CUBE_POS_Z;
    let col_base = col_block * TX;
    let col = col_base + tx;
    let in_bounds = col < width;

    let w = width as usize;
    let pt = pad_total as usize;
    let ch_base = (channel as usize) * pt;
    let period_x = 2u32 * (width - 1u32);
    let period_y = 2u32 * (height - 1u32);

    let strip_h_base = height / n_strips;
    let strip_rem = height - strip_h_base * n_strips;
    let y_start = strip * strip_h_base + u32::min(strip, strip_rem);
    let y_end_unclamp = y_start + strip_h_base + (if strip < strip_rem { 1u32 } else { 0u32 });
    let y_end = u32::min(y_end_unclamp, height);

    let mut src_row = SharedMemory::<f32>::new(TILE_COLS_US);
    let mut dst_row = SharedMemory::<f32>::new(TILE_COLS_US);
    let mut buf_mu1 = SharedMemory::<f32>::new(BUF_LEN_US);
    let mut buf_mu2 = SharedMemory::<f32>::new(BUF_LEN_US);
    let mut buf_sq = SharedMemory::<f32>::new(BUF_LEN_US);
    let mut buf_s12 = SharedMemory::<f32>::new(BUF_LEN_US);

    let mut sum_m1 = 0.0_f32;
    let mut sum_m2 = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut sum_s12 = 0.0_f32;

    // ============================ PREFIX INIT ============================
    let mut k: u32 = 0u32;
    while k < DIAM {
        let raw_y = (y_start + k + period_y - R) % period_y;
        let y_in = if raw_y < height {
            raw_y
        } else {
            period_y - raw_y
        };

        sync_cube();
        let mut i: u32 = 0u32;
        while i * TX + tx < TILE_COLS {
            let load_x = i * TX + tx;
            let raw_x = (col_base + load_x + period_x - R) % period_x;
            let gx = if raw_x < width {
                raw_x
            } else {
                period_x - raw_x
            };
            let off = (y_in as usize) * w + (gx as usize);
            let s_val = if channel == 0u32 {
                src_a[off]
            } else if channel == 1u32 {
                src_b[off]
            } else {
                src_c[off]
            };
            let d_val = if channel == 0u32 {
                dst_a[off]
            } else if channel == 1u32 {
                dst_b[off]
            } else {
                dst_c[off]
            };
            src_row[load_x as usize] = s_val;
            dst_row[load_x as usize] = d_val;
            i += 1u32;
        }
        sync_cube();

        let mut m1 = 0.0_f32;
        let mut m2 = 0.0_f32;
        let mut sq = 0.0_f32;
        let mut s12 = 0.0_f32;
        let mut j: u32 = 0u32;
        while j < DIAM {
            let s = src_row[(tx + j) as usize];
            let d = dst_row[(tx + j) as usize];
            m1 += s;
            m2 += d;
            sq = fma(s, s, fma(d, d, sq));
            s12 = fma(s, d, s12);
            j += 1u32;
        }
        m1 *= INV_DIAM;
        m2 *= INV_DIAM;
        sq *= INV_DIAM;
        s12 *= INV_DIAM;

        let buf_idx = (k * TX + tx) as usize;
        buf_mu1[buf_idx] = m1;
        buf_mu2[buf_idx] = m2;
        buf_sq[buf_idx] = sq;
        buf_s12[buf_idx] = s12;

        sum_m1 += m1;
        sum_m2 += m2;
        sum_sq += sq;
        sum_s12 += s12;

        k += 1u32;
    }

    // ============================ WALK Y ============================
    let mut slot: u32 = 0u32;
    let mut y: u32 = y_start;
    while y < y_end {
        let mu1 = sum_m1 * INV_DIAM;
        let mu2 = sum_m2 * INV_DIAM;
        let ssq = sum_sq * INV_DIAM;
        let s12_v = sum_s12 * INV_DIAM;

        let off = (y as usize) * w + (col as usize);
        if in_bounds {
            mu1_all[ch_base + off] = mu1;
            mu2_all[ch_base + off] = mu2;
            ssq_all[ch_base + off] = ssq;
            s12_all[ch_base + off] = s12_v;
        }

        // Slide
        let buf_idx = (slot * TX + tx) as usize;
        let old_m1 = buf_mu1[buf_idx];
        let old_m2 = buf_mu2[buf_idx];
        let old_sq = buf_sq[buf_idx];
        let old_s12 = buf_s12[buf_idx];

        let raw_y = (y + R + 1u32 + period_y) % period_y;
        let y_in = if raw_y < height {
            raw_y
        } else {
            period_y - raw_y
        };

        sync_cube();
        let mut i: u32 = 0u32;
        while i * TX + tx < TILE_COLS {
            let load_x = i * TX + tx;
            let raw_x = (col_base + load_x + period_x - R) % period_x;
            let gx = if raw_x < width {
                raw_x
            } else {
                period_x - raw_x
            };
            let off2 = (y_in as usize) * w + (gx as usize);
            let s_val = if channel == 0u32 {
                src_a[off2]
            } else if channel == 1u32 {
                src_b[off2]
            } else {
                src_c[off2]
            };
            let d_val = if channel == 0u32 {
                dst_a[off2]
            } else if channel == 1u32 {
                dst_b[off2]
            } else {
                dst_c[off2]
            };
            src_row[load_x as usize] = s_val;
            dst_row[load_x as usize] = d_val;
            i += 1u32;
        }
        sync_cube();

        let mut nm1 = 0.0_f32;
        let mut nm2 = 0.0_f32;
        let mut nsq = 0.0_f32;
        let mut ns12 = 0.0_f32;
        let mut j: u32 = 0u32;
        while j < DIAM {
            let s = src_row[(tx + j) as usize];
            let d = dst_row[(tx + j) as usize];
            nm1 += s;
            nm2 += d;
            nsq = fma(s, s, fma(d, d, nsq));
            ns12 = fma(s, d, ns12);
            j += 1u32;
        }
        nm1 *= INV_DIAM;
        nm2 *= INV_DIAM;
        nsq *= INV_DIAM;
        ns12 *= INV_DIAM;

        sum_m1 = sum_m1 + nm1 - old_m1;
        sum_m2 = sum_m2 + nm2 - old_m2;
        sum_sq = sum_sq + nsq - old_sq;
        sum_s12 = sum_s12 + ns12 - old_s12;

        buf_mu1[buf_idx] = nm1;
        buf_mu2[buf_idx] = nm2;
        buf_sq[buf_idx] = nsq;
        buf_s12[buf_idx] = ns12;

        slot = (slot + 1u32) % DIAM;
        y += 1u32;
    }
}

// ───────────── the real producer (verbatim from zensim-gpu fused.rs:489) ─────────────
// Tile-fused H-blur + V-blur + per-pixel features, writing the mu1/mu2/ssq/s12
// persist planes that `per_scale` later consumes. Grid: (ceil(w/TX), n_strips, 3),
// channel = CUBE_POS_Z. Kept byte-identical to the shipping kernel so the repro
// exercises the same MSL the metric ships.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn fused_features_kernel_persist(
    src_a: &Array<f32>,
    dst_a: &Array<f32>,
    src_b: &Array<f32>,
    dst_b: &Array<f32>,
    src_c: &Array<f32>,
    dst_c: &Array<f32>,
    partials_f64: &mut Array<f64>,
    partials_max: &mut Array<f32>,
    mu1_all: &mut Array<f32>,
    mu2_all: &mut Array<f32>,
    ssq_all: &mut Array<f32>,
    s12_all: &mut Array<f32>,
    width: u32, // padded_w
    height: u32,
    n_strips: u32,
    slot_off_f64: u32,
    slot_off_max: u32,
    pad_total: u32,
    y_body_start: u32,
    y_body_end: u32,
) {
    let tx = UNIT_POS_X;
    let col_block = CUBE_POS_X;
    let strip = CUBE_POS_Y;
    let channel = CUBE_POS_Z;
    let col_base = col_block * TX;
    let col = col_base + tx;
    let in_bounds = col < width;

    let w = width as usize;
    let n_strips_us = n_strips as usize;
    let pw = width as usize;
    let pt = pad_total as usize;
    let ch_base = (channel as usize) * pt;
    let period_x = 2u32 * (width - 1u32);
    let period_y = 2u32 * (height - 1u32);

    let strip_h_base = height / n_strips;
    let strip_rem = height - strip_h_base * n_strips;
    let y_start = strip * strip_h_base + u32::min(strip, strip_rem);
    let y_end_unclamp = y_start + strip_h_base + (if strip < strip_rem { 1u32 } else { 0u32 });
    let y_end = u32::min(y_end_unclamp, height);

    let mut src_row = SharedMemory::<f32>::new(TILE_COLS_US);
    let mut dst_row = SharedMemory::<f32>::new(TILE_COLS_US);
    let mut buf_mu1 = SharedMemory::<f32>::new(BUF_LEN_US);
    let mut buf_mu2 = SharedMemory::<f32>::new(BUF_LEN_US);
    let mut buf_sq = SharedMemory::<f32>::new(BUF_LEN_US);
    let mut buf_s12 = SharedMemory::<f32>::new(BUF_LEN_US);

    let mut sum_m1 = 0.0_f32;
    let mut sum_m2 = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut sum_s12 = 0.0_f32;

    let mut a0 = 0.0_f64;
    let mut a1 = 0.0_f64;
    let mut a2 = 0.0_f64;
    let mut a3 = 0.0_f64;
    let mut a4 = 0.0_f64;
    let mut a5 = 0.0_f64;
    let mut a6 = 0.0_f64;
    let mut a7 = 0.0_f64;
    let mut a8 = 0.0_f64;
    let mut a9 = 0.0_f64;
    let mut a10 = 0.0_f64;
    let mut a11 = 0.0_f64;
    let mut a12 = 0.0_f64;
    let mut a13 = 0.0_f64;
    let mut a14 = 0.0_f64;
    let mut a15 = 0.0_f64;
    let mut a16 = 0.0_f64;
    let mut peak0 = 0.0_f32;
    let mut peak1 = 0.0_f32;
    let mut peak2 = 0.0_f32;

    // ============================ PREFIX INIT ============================
    let mut k: u32 = 0u32;
    while k < DIAM {
        let raw_y = (y_start + k + period_y - R) % period_y;
        let y_in = if raw_y < height {
            raw_y
        } else {
            period_y - raw_y
        };

        sync_cube();
        let mut i: u32 = 0u32;
        while i * TX + tx < TILE_COLS {
            let load_x = i * TX + tx;
            let raw_x = (col_base + load_x + period_x - R) % period_x;
            let gx = if raw_x < width {
                raw_x
            } else {
                period_x - raw_x
            };
            let off = (y_in as usize) * w + (gx as usize);
            let s_val = if channel == 0u32 {
                src_a[off]
            } else if channel == 1u32 {
                src_b[off]
            } else {
                src_c[off]
            };
            let d_val = if channel == 0u32 {
                dst_a[off]
            } else if channel == 1u32 {
                dst_b[off]
            } else {
                dst_c[off]
            };
            src_row[load_x as usize] = s_val;
            dst_row[load_x as usize] = d_val;
            i += 1u32;
        }
        sync_cube();

        let mut m1 = 0.0_f32;
        let mut m2 = 0.0_f32;
        let mut sq = 0.0_f32;
        let mut s12 = 0.0_f32;
        let mut j: u32 = 0u32;
        while j < DIAM {
            let s = src_row[(tx + j) as usize];
            let d = dst_row[(tx + j) as usize];
            m1 += s;
            m2 += d;
            sq = fma(s, s, fma(d, d, sq));
            s12 = fma(s, d, s12);
            j += 1u32;
        }
        m1 *= INV_DIAM;
        m2 *= INV_DIAM;
        sq *= INV_DIAM;
        s12 *= INV_DIAM;

        let buf_idx = (k * TX + tx) as usize;
        buf_mu1[buf_idx] = m1;
        buf_mu2[buf_idx] = m2;
        buf_sq[buf_idx] = sq;
        buf_s12[buf_idx] = s12;

        sum_m1 += m1;
        sum_m2 += m2;
        sum_sq += sq;
        sum_s12 += s12;

        k += 1u32;
    }

    // ============================ WALK Y ============================
    let mut slot: u32 = 0u32;
    let mut y: u32 = y_start;
    while y < y_end {
        let mu1 = sum_m1 * INV_DIAM;
        let mu2 = sum_m2 * INV_DIAM;
        let ssq = sum_sq * INV_DIAM;
        let s12_v = sum_s12 * INV_DIAM;

        let off = (y as usize) * w + (col as usize);
        if in_bounds {
            mu1_all[ch_base + off] = mu1;
            mu2_all[ch_base + off] = mu2;
            ssq_all[ch_base + off] = ssq;
            s12_all[ch_base + off] = s12_v;
        }

        let mut sv: f32 = 0.0;
        let mut dv: f32 = 0.0;
        if in_bounds {
            if channel == 0u32 {
                sv = src_a[off];
                dv = dst_a[off];
            } else {
                if channel == 1u32 {
                    sv = src_b[off];
                    dv = dst_b[off];
                } else {
                    sv = src_c[off];
                    dv = dst_c[off];
                }
            }
        }

        let mu_diff = mu1 - mu2;
        let num_m = fma(mu_diff, -mu_diff, 1.0);
        let inner_ns = fma(-mu1, mu2, s12_v);
        let num_s = fma(2.0, inner_ns, C2);
        let inner_ds_inner = fma(-mu1, mu1, ssq);
        let denom_s = fma(-mu2, mu2, inner_ds_inner) + C2;
        let sd_raw = 1.0 - (num_m * num_s) / denom_s;
        let sd0 = if sd_raw > 0.0 { sd_raw } else { f32::new(0.0) };
        let is_body = y >= y_body_start && y < y_body_end;
        let mask = if is_body {
            f32::new(1.0)
        } else {
            f32::new(0.0)
        };
        let sd = sd0 * mask;
        let sd2 = sd * sd;
        let sd4 = sd2 * sd2;
        a0 += sd as f64;
        a1 += sd4 as f64;
        a2 += sd2 as f64;
        a14 += (sd4 * sd4) as f64;
        if sd > peak0 {
            peak0 = sd;
        }

        let diff1 = f32::abs(sv - mu1);
        let diff2 = f32::abs(dv - mu2);
        let ed = (1.0 + diff2) / (1.0 + diff1) - 1.0;
        let artifact0 = if ed > 0.0 { ed } else { f32::new(0.0) };
        let detail_lost0 = if ed < 0.0 { -ed } else { f32::new(0.0) };
        let artifact = artifact0 * mask;
        let detail_lost = detail_lost0 * mask;
        let a2_v = artifact * artifact;
        let dl2 = detail_lost * detail_lost;
        let a4_v = a2_v * a2_v;
        let dl4 = dl2 * dl2;
        a3 += artifact as f64;
        a4 += a4_v as f64;
        a5 += a2_v as f64;
        a6 += detail_lost as f64;
        a7 += dl4 as f64;
        a8 += dl2 as f64;
        a15 += (a4_v * a4_v) as f64;
        a16 += (dl4 * dl4) as f64;
        if artifact > peak1 {
            peak1 = artifact;
        }
        if detail_lost > peak2 {
            peak2 = detail_lost;
        }

        let vs = (sv - mu1) * mask;
        let vd = (dv - mu2) * mask;
        a10 += (vs * vs) as f64;
        a11 += (vd * vd) as f64;
        a12 += (diff1 * mask) as f64;
        a13 += (diff2 * mask) as f64;

        let pd = (sv - dv) * mask;
        a9 += (pd * pd) as f64;

        // Slide
        let buf_idx = (slot * TX + tx) as usize;
        let old_m1 = buf_mu1[buf_idx];
        let old_m2 = buf_mu2[buf_idx];
        let old_sq = buf_sq[buf_idx];
        let old_s12 = buf_s12[buf_idx];

        let raw_y = (y + R + 1u32 + period_y) % period_y;
        let y_in = if raw_y < height {
            raw_y
        } else {
            period_y - raw_y
        };

        sync_cube();
        let mut i: u32 = 0u32;
        while i * TX + tx < TILE_COLS {
            let load_x = i * TX + tx;
            let raw_x = (col_base + load_x + period_x - R) % period_x;
            let gx = if raw_x < width {
                raw_x
            } else {
                period_x - raw_x
            };
            let off2 = (y_in as usize) * w + (gx as usize);
            let s_val = if channel == 0u32 {
                src_a[off2]
            } else if channel == 1u32 {
                src_b[off2]
            } else {
                src_c[off2]
            };
            let d_val = if channel == 0u32 {
                dst_a[off2]
            } else if channel == 1u32 {
                dst_b[off2]
            } else {
                dst_c[off2]
            };
            src_row[load_x as usize] = s_val;
            dst_row[load_x as usize] = d_val;
            i += 1u32;
        }
        sync_cube();

        let mut nm1 = 0.0_f32;
        let mut nm2 = 0.0_f32;
        let mut nsq = 0.0_f32;
        let mut ns12 = 0.0_f32;
        let mut j: u32 = 0u32;
        while j < DIAM {
            let s = src_row[(tx + j) as usize];
            let d = dst_row[(tx + j) as usize];
            nm1 += s;
            nm2 += d;
            nsq = fma(s, s, fma(d, d, nsq));
            ns12 = fma(s, d, ns12);
            j += 1u32;
        }
        nm1 *= INV_DIAM;
        nm2 *= INV_DIAM;
        nsq *= INV_DIAM;
        ns12 *= INV_DIAM;

        sum_m1 = sum_m1 + nm1 - old_m1;
        sum_m2 = sum_m2 + nm2 - old_m2;
        sum_sq = sum_sq + nsq - old_sq;
        sum_s12 = sum_s12 + ns12 - old_s12;

        buf_mu1[buf_idx] = nm1;
        buf_mu2[buf_idx] = nm2;
        buf_sq[buf_idx] = nsq;
        buf_s12[buf_idx] = ns12;

        slot = (slot + 1u32) % DIAM;
        y += 1u32;
    }

    if !in_bounds {
        terminate!();
    }
    let slot_idx_us =
        (channel as usize) * n_strips_us * pw + (strip as usize) * pw + (col as usize);
    let f64_base = (slot_off_f64 as usize) + slot_idx_us * 17;
    partials_f64[f64_base] = a0;
    partials_f64[f64_base + 1] = a1;
    partials_f64[f64_base + 2] = a2;
    partials_f64[f64_base + 3] = a3;
    partials_f64[f64_base + 4] = a4;
    partials_f64[f64_base + 5] = a5;
    partials_f64[f64_base + 6] = a6;
    partials_f64[f64_base + 7] = a7;
    partials_f64[f64_base + 8] = a8;
    partials_f64[f64_base + 9] = a9;
    partials_f64[f64_base + 10] = a10;
    partials_f64[f64_base + 11] = a11;
    partials_f64[f64_base + 12] = a12;
    partials_f64[f64_base + 13] = a13;
    partials_f64[f64_base + 14] = a14;
    partials_f64[f64_base + 15] = a15;
    partials_f64[f64_base + 16] = a16;
    let max_base = (slot_off_max as usize) + slot_idx_us * 3;
    partials_max[max_base] = peak0;
    partials_max[max_base + 1] = peak1;
    partials_max[max_base + 2] = peak2;
}

// ───────────────────────── CPU reference (plain Rust) ─────────────────────────

fn per_pixel_ssim_error(mu1: f32, mu2: f32, ssq: f32, s12: f32) -> f32 {
    let c2: f32 = 0.0009;
    let mu_diff = mu1 - mu2;
    let num_m = mu_diff.mul_add(-mu_diff, 1.0);
    let inner_ns = (-mu1).mul_add(mu2, s12);
    let num_s = 2.0_f32.mul_add(inner_ns, c2);
    let inner_ds = (-mu1).mul_add(mu1, ssq);
    let denom_s = (-mu2).mul_add(mu2, inner_ds) + c2;
    let sd_raw = 1.0 - (num_m * num_s) / denom_s;
    if sd_raw > 0.0 { sd_raw } else { 0.0 }
}

#[allow(clippy::too_many_arguments)]
fn cpu_per_scale(
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    pad_total: usize,
    n: usize, // padded_w * height
    w: [f32; 3],
) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (idx, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for c in 0..3 {
            let o = idx + c * pad_total;
            acc += w[c] * per_pixel_ssim_error(mu1[o], mu2[o], ssq[o], s12[o]);
        }
        *slot = acc;
    }
    out
}

fn cpu_upsample_add(
    src: &[f32],
    src_w: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
    log2_factor: u32,
    blend: f32,
) {
    for idx in 0..dst_w * dst_h {
        let dx = idx % dst_w;
        let dy = idx / dst_w;
        let sx = (dx >> log2_factor).min(src_w.saturating_sub(1));
        let sy = dy >> log2_factor; // src_h-1 clamp folded in below
        let v = src[sy * src_w + sx];
        dst[idx] += blend * v;
    }
}

// ───────── CPU reference for the feature kernel's blurred SSIM moments ─────────
// Centered 11×11 mirror box blur (reflect-101, period 2*(dim-1)) over the padded
// buffer, matching `fused_features_kernel_persist`. Computed in f64 for accuracy;
// compared with a 1e-3 tolerance, so fma-order differences are irrelevant.

fn mirror_idx(idx: i64, dim: i64) -> usize {
    let period = 2 * (dim - 1);
    let mut r = idx % period;
    if r < 0 {
        r += period;
    }
    (if r < dim { r } else { period - r }) as usize
}

/// Produce the channel-concatenated [ch0|ch1|ch2] mu1/mu2/ssq/s12 planes
/// (each `pt*3` long) from per-channel src/dst XYB planes (`pt` long each).
fn cpu_feature_planes(
    src: &[Vec<f32>; 3],
    dst: &[Vec<f32>; 3],
    pw: usize,
    ph: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let pt = pw * ph;
    let mut mu1 = vec![0.0f32; pt * 3];
    let mut mu2 = vec![0.0f32; pt * 3];
    let mut ssq = vec![0.0f32; pt * 3];
    let mut s12 = vec![0.0f32; pt * 3];
    let r = R as i64;
    let diam = DIAM as i64;
    let inv = 1.0f64 / (DIAM as f64 * DIAM as f64);
    for c in 0..3 {
        let sc = &src[c];
        let dc = &dst[c];
        for y in 0..ph {
            for x in 0..pw {
                let mut acc_m1 = 0.0f64;
                let mut acc_m2 = 0.0f64;
                let mut acc_sq = 0.0f64;
                let mut acc_s12 = 0.0f64;
                for ky in 0..diam {
                    let sy = mirror_idx(y as i64 + ky - r, ph as i64);
                    for kx in 0..diam {
                        let sx = mirror_idx(x as i64 + kx - r, pw as i64);
                        let s = sc[sy * pw + sx] as f64;
                        let d = dc[sy * pw + sx] as f64;
                        acc_m1 += s;
                        acc_m2 += d;
                        acc_sq += s * s + d * d;
                        acc_s12 += s * d;
                    }
                }
                let off = c * pt + y * pw + x;
                mu1[off] = (acc_m1 * inv) as f32;
                mu2[off] = (acc_m2 * inv) as f32;
                ssq[off] = (acc_sq * inv) as f32;
                s12[off] = (acc_s12 * inv) as f32;
            }
        }
    }
    (mu1, mu2, ssq, s12)
}

/// Mirrors `zensim_gpu::pick_n_strips`.
fn pick_n_strips(padded_w: u32, height: u32) -> u32 {
    if height <= 64 {
        1
    } else if height >= 1024 {
        8
    } else if padded_w >= 256 {
        4
    } else {
        2
    }
}

/// Deterministic synthetic XYB plane value (host-computed, then uploaded — the
/// feature kernel's inputs being clean/host-uploaded is exactly what isolates a
/// *producer* codegen bug from its upstream input kernels).
fn synth_xyb(scale: usize, channel: usize, side: usize, x: usize, y: usize) -> f32 {
    let fx = x as f32;
    let fy = y as f32;
    let base = 0.3 + 0.2 * ((scale * 3 + channel * 5) as f32 + fx * 0.05 + fy * 0.07).sin();
    let dist = if side == 1 {
        0.03 * (fx * 0.11 + fy * 0.09).cos()
    } else {
        0.0
    };
    base + dist
}

// ───────────────────────── harness ─────────────────────────

const N_SCALES: usize = 4;

/// How the mu1/mu2/ssq/s12 planes that `per_scale` reads get filled.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Producer {
    /// Host blit (`create_from_slice`) — the original chain. CI-confirmed PASS on Metal.
    Host,
    /// On-device 1D flat relay copy from host-staged temps into the planes.
    Relay1d,
    /// On-device 3D (channel-as-z) relay — mirrors the feature kernel's grid shape.
    Relay3d,
}

impl Producer {
    fn label(self) -> &'static str {
        match self {
            Producer::Host => "host-upload",
            Producer::Relay1d => "gpu-relay-1d",
            Producer::Relay3d => "gpu-relay-3d(ch=z)",
        }
    }
}

const RELAY_TX: u32 = 256;

/// Round up to a multiple of 16 (mirrors zensim_gpu::simd_padded_width for the
/// sizes used here; both 64 and 96 are already multiples of 16).
fn simd_padded_width(w: usize) -> usize {
    (w + 15) & !15
}

fn cube_count(n: usize) -> CubeCount {
    CubeCount::Static((n as u32).div_ceil(256).max(1), 1, 1)
}

/// Deterministic synthetic plane value — structured so `sd_raw` straddles 0
/// (exercises the `max(0, ·)` clamp) and varies across the buffer.
fn synth(scale: usize, plane: usize, idx: usize, pt: usize) -> f32 {
    let ch = idx / pt;
    let i = (idx % pt) as f32;
    let base = 0.4 + 0.15 * ((scale * 7 + plane * 3 + ch * 5) as f32).sin();
    base + 0.01 * (i * 0.013 + plane as f32).sin()
}

fn run<R: Runtime>(
    label: &str,
    width: usize,
    height: usize,
    zero_fill: bool,
    producer: Producer,
) -> bool {
    let client = R::client(&Default::default());

    // Build the pyramid plan (padded_w halves via /2; logical w via div_ceil;
    // height via div_ceil) — matches zensim-gpu's scale build.
    let mut padded_w = simd_padded_width(width);
    let mut h = height;
    let mut plan: Vec<(usize, usize)> = Vec::new(); // (padded_w, height) per scale
    for _ in 0..N_SCALES {
        if padded_w < 8 || h < 8 {
            break;
        }
        plan.push((padded_w, h));
        padded_w /= 2;
        h = h.div_ceil(2);
    }

    let base_pw = plan[0].0;
    let base_n = base_pw * height;
    let w = [1.0f32 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let blend = 1.0f32 / plan.len() as f32;

    // GPU accumulator (zero-filled).
    let acc = client.empty(base_n * core::mem::size_of::<f32>());
    unsafe {
        diffmap_zero_kernel::launch_unchecked::<R>(
            &client,
            cube_count(base_n),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(acc.clone(), base_n),
            base_n as u32,
        );
    }

    // CPU accumulator.
    let mut acc_cpu = vec![0.0f32; base_n];

    for (s, &(pw, ph)) in plan.iter().enumerate() {
        let pt = pw * ph; // pad_total
        let plane_len = pt * 3;

        // Synthetic mu1/mu2/ssq/s12 (channel-concatenated [ch0|ch1|ch2]).
        let mk =
            |plane: usize| -> Vec<f32> { (0..plane_len).map(|i| synth(s, plane, i, pt)).collect() };
        let mu1 = mk(0);
        let mu2 = mk(1);
        let ssq = mk(2);
        let s12 = mk(3);

        // The four planes `per_scale` will read. For `Host` they ARE the
        // host-blitted buffers. For relay modes they are empty device buffers
        // that a prior GPU dispatch fills from host-staged temps — so the read
        // in `per_scale` is a genuine producer→consumer storage read-after-write.
        let (mu1_h, mu2_h, ssq_h, s12_h) = match producer {
            Producer::Host => (
                client.create_from_slice(f32::as_bytes(&mu1)),
                client.create_from_slice(f32::as_bytes(&mu2)),
                client.create_from_slice(f32::as_bytes(&ssq)),
                client.create_from_slice(f32::as_bytes(&s12)),
            ),
            Producer::Relay1d | Producer::Relay3d => {
                let t1 = client.create_from_slice(f32::as_bytes(&mu1));
                let t2 = client.create_from_slice(f32::as_bytes(&mu2));
                let t3 = client.create_from_slice(f32::as_bytes(&ssq));
                let t4 = client.create_from_slice(f32::as_bytes(&s12));
                let p1 = client.empty(plane_len * core::mem::size_of::<f32>());
                let p2 = client.empty(plane_len * core::mem::size_of::<f32>());
                let p3 = client.empty(plane_len * core::mem::size_of::<f32>());
                let p4 = client.empty(plane_len * core::mem::size_of::<f32>());
                if producer == Producer::Relay1d {
                    unsafe {
                        relay_planes_1d_kernel::launch_unchecked::<R>(
                            &client,
                            cube_count(plane_len),
                            CubeDim::new_1d(256),
                            ArrayArg::from_raw_parts(t1.clone(), plane_len),
                            ArrayArg::from_raw_parts(t2.clone(), plane_len),
                            ArrayArg::from_raw_parts(t3.clone(), plane_len),
                            ArrayArg::from_raw_parts(t4.clone(), plane_len),
                            ArrayArg::from_raw_parts(p1.clone(), plane_len),
                            ArrayArg::from_raw_parts(p2.clone(), plane_len),
                            ArrayArg::from_raw_parts(p3.clone(), plane_len),
                            ArrayArg::from_raw_parts(p4.clone(), plane_len),
                            plane_len as u32,
                        );
                    }
                } else {
                    let cube_x = (pt as u32).div_ceil(RELAY_TX).max(1);
                    unsafe {
                        relay_planes_3d_kernel::launch_unchecked::<R>(
                            &client,
                            CubeCount::Static(cube_x, 1, 3),
                            CubeDim::new_3d(RELAY_TX, 1, 1),
                            ArrayArg::from_raw_parts(t1.clone(), plane_len),
                            ArrayArg::from_raw_parts(t2.clone(), plane_len),
                            ArrayArg::from_raw_parts(t3.clone(), plane_len),
                            ArrayArg::from_raw_parts(t4.clone(), plane_len),
                            ArrayArg::from_raw_parts(p1.clone(), plane_len),
                            ArrayArg::from_raw_parts(p2.clone(), plane_len),
                            ArrayArg::from_raw_parts(p3.clone(), plane_len),
                            ArrayArg::from_raw_parts(p4.clone(), plane_len),
                            pt as u32,
                        );
                    }
                }
                (p1, p2, p3, p4)
            }
        };

        let scale_dm = client.empty(pt * core::mem::size_of::<f32>());

        // Step 1 — optional defensive zero-fill of scale_dm (zenmetrics 648a8c7b).
        if zero_fill {
            unsafe {
                diffmap_zero_kernel::launch_unchecked::<R>(
                    &client,
                    cube_count(pt),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(scale_dm.clone(), pt),
                    pt as u32,
                );
            }
        }

        // Step 2 — per-scale weighted SSIM error -> scale_dm.
        unsafe {
            per_scale_weighted_ssim_kernel::launch_unchecked::<R>(
                &client,
                cube_count(pt),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(mu1_h.clone(), plane_len),
                ArrayArg::from_raw_parts(mu2_h.clone(), plane_len),
                ArrayArg::from_raw_parts(ssq_h.clone(), plane_len),
                ArrayArg::from_raw_parts(s12_h.clone(), plane_len),
                ArrayArg::from_raw_parts(scale_dm.clone(), pt),
                pw as u32,
                ph as u32,
                pt as u32,
                w[0],
                w[1],
                w[2],
            );
        }

        // Step 3 — upsample-add scale_dm (x2^s) into acc.
        unsafe {
            pow2x_upsample_add_kernel::launch_unchecked::<R>(
                &client,
                cube_count(base_n),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(scale_dm.clone(), pt),
                ArrayArg::from_raw_parts(acc.clone(), base_n),
                pw as u32,
                ph as u32,
                base_pw as u32,
                height as u32,
                s as u32,
                blend,
            );
        }

        // CPU mirror.
        let dm_cpu = cpu_per_scale(&mu1, &mu2, &ssq, &s12, pt, pt, w);
        cpu_upsample_add(&dm_cpu, pw, &mut acc_cpu, base_pw, height, s as u32, blend);
    }

    // Read back + compare.
    let bytes = client.read_one(acc.clone()).expect("read acc");
    let gpu = f32::from_bytes(&bytes);

    let mut max_err = 0.0f32;
    let mut argmax = 0usize;
    let mut n_div = 0usize;
    for i in 0..base_n {
        let e = (gpu[i] - acc_cpu[i]).abs();
        if e > 1e-3 {
            n_div += 1;
        }
        if e > max_err {
            max_err = e;
            argmax = i;
        }
    }
    let pass = max_err <= 1e-3;
    println!(
        "  [{:>18}] {label} ({width}x{height}, base_pw={base_pw}): max_err = {max_err:.6} \
         ({n_div}/{base_n} px > 1e-3); argmax (x={}, y={}) gpu={} cpu={}  -> {}",
        producer.label(),
        argmax % base_pw,
        argmax / base_pw,
        gpu[argmax],
        acc_cpu[argmax],
        if pass { "PASS" } else { "FAIL" },
    );
    if !pass {
        // Show the first few divergent pixels (the scattered "stuck" values).
        let mut shown = 0;
        for i in 0..base_n {
            if (gpu[i] - acc_cpu[i]).abs() > 1e-3 {
                println!(
                    "      (x={}, y={}) gpu={:.6} cpu={:.6}",
                    i % base_pw,
                    i / base_pw,
                    gpu[i],
                    acc_cpu[i]
                );
                shown += 1;
                if shown >= 8 {
                    break;
                }
            }
        }
    }
    pass
}

/// The faithful-producer experiment: the mu1/mu2/ssq/s12 planes are produced by
/// the REAL `fused_features_kernel_persist` (from clean host-uploaded XYB), read
/// back and compared to a CPU blur reference, then fed through per_scale+upsample.
/// Reports BOTH the persist-plane divergence (producer correctness) and the final
/// acc divergence (full chain). This is the HANDOFF's decisive experiment.
fn run_feature<R: Runtime>(label: &str, width: usize, height: usize, mode: FeatMode) -> bool {
    let client = R::client(&Default::default());

    let mut padded_w = simd_padded_width(width);
    let mut h = height;
    let mut plan: Vec<(usize, usize)> = Vec::new();
    for _ in 0..N_SCALES {
        if padded_w < 8 || h < 8 {
            break;
        }
        plan.push((padded_w, h));
        padded_w /= 2;
        h = h.div_ceil(2);
    }

    let base_pw = plan[0].0;
    let base_n = base_pw * height;
    let w = [1.0f32 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let blend = 1.0f32 / plan.len() as f32;

    let acc = client.empty(base_n * core::mem::size_of::<f32>());
    unsafe {
        diffmap_zero_kernel::launch_unchecked::<R>(
            &client,
            cube_count(base_n),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(acc.clone(), base_n),
            base_n as u32,
        );
    }
    let mut acc_cpu = vec![0.0f32; base_n];

    let mut worst_plane_err = 0.0f32;
    let mut worst_plane_where = (0usize, 0usize, 0usize, 0usize); // (scale, channel, x, y)
    let mut plane_div_total = 0usize;

    for (s, &(pw, ph)) in plan.iter().enumerate() {
        let pt = pw * ph;
        let plane_len = pt * 3;
        let n_strips = pick_n_strips(pw as u32, ph as u32);

        // Synthetic per-channel XYB inputs (ref = side 0, dist = side 1).
        let mk = |channel: usize, side: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; pt];
            for y in 0..ph {
                for x in 0..pw {
                    v[y * pw + x] = synth_xyb(s, channel, side, x, y);
                }
            }
            v
        };
        let src: [Vec<f32>; 3] = [mk(0, 0), mk(1, 0), mk(2, 0)];
        let dst: [Vec<f32>; 3] = [mk(0, 1), mk(1, 1), mk(2, 1)];

        let src_a = client.create_from_slice(f32::as_bytes(&src[0]));
        let dst_a = client.create_from_slice(f32::as_bytes(&dst[0]));
        let src_b = client.create_from_slice(f32::as_bytes(&src[1]));
        let dst_b = client.create_from_slice(f32::as_bytes(&dst[1]));
        let src_c = client.create_from_slice(f32::as_bytes(&src[2]));
        let dst_c = client.create_from_slice(f32::as_bytes(&dst[2]));

        // Partials buffers (kept verbatim with the kernel even though we don't read them).
        let n_partials_f64 = pw * (n_strips as usize) * 3 * 17;
        let n_partials_max = pw * (n_strips as usize) * 3 * 3;
        let partials_f64 = client.empty(n_partials_f64 * core::mem::size_of::<f64>());
        let partials_max = client.empty(n_partials_max * core::mem::size_of::<f32>());

        let p_mu1 = client.empty(plane_len * core::mem::size_of::<f32>());
        let p_mu2 = client.empty(plane_len * core::mem::size_of::<f32>());
        let p_ssq = client.empty(plane_len * core::mem::size_of::<f32>());
        let p_s12 = client.empty(plane_len * core::mem::size_of::<f32>());

        let cube_x = (pw as u32).div_ceil(TX).max(1);
        if mode == FeatMode::BlurOnly {
            unsafe {
                blur_planes_kernel::launch_unchecked::<R>(
                    &client,
                    CubeCount::Static(cube_x, n_strips, 3),
                    CubeDim::new_3d(TX, 1, 1),
                    ArrayArg::from_raw_parts(src_a.clone(), pt),
                    ArrayArg::from_raw_parts(dst_a.clone(), pt),
                    ArrayArg::from_raw_parts(src_b.clone(), pt),
                    ArrayArg::from_raw_parts(dst_b.clone(), pt),
                    ArrayArg::from_raw_parts(src_c.clone(), pt),
                    ArrayArg::from_raw_parts(dst_c.clone(), pt),
                    ArrayArg::from_raw_parts(p_mu1.clone(), plane_len),
                    ArrayArg::from_raw_parts(p_mu2.clone(), plane_len),
                    ArrayArg::from_raw_parts(p_ssq.clone(), plane_len),
                    ArrayArg::from_raw_parts(p_s12.clone(), plane_len),
                    pw as u32,
                    ph as u32,
                    n_strips,
                    pt as u32,
                );
            }
        } else if mode == FeatMode::BlurF64 {
            let n_pf64 = pw * (n_strips as usize) * 3;
            let pf64 = client.empty(n_pf64 * core::mem::size_of::<f64>());
            unsafe {
                blur_f64_kernel::launch_unchecked::<R>(
                    &client,
                    CubeCount::Static(cube_x, n_strips, 3),
                    CubeDim::new_3d(TX, 1, 1),
                    ArrayArg::from_raw_parts(src_a.clone(), pt),
                    ArrayArg::from_raw_parts(dst_a.clone(), pt),
                    ArrayArg::from_raw_parts(src_b.clone(), pt),
                    ArrayArg::from_raw_parts(dst_b.clone(), pt),
                    ArrayArg::from_raw_parts(src_c.clone(), pt),
                    ArrayArg::from_raw_parts(dst_c.clone(), pt),
                    ArrayArg::from_raw_parts(pf64.clone(), n_pf64),
                    ArrayArg::from_raw_parts(p_mu1.clone(), plane_len),
                    ArrayArg::from_raw_parts(p_mu2.clone(), plane_len),
                    ArrayArg::from_raw_parts(p_ssq.clone(), plane_len),
                    ArrayArg::from_raw_parts(p_s12.clone(), plane_len),
                    pw as u32,
                    ph as u32,
                    n_strips,
                    pt as u32,
                );
            }
        } else {
            unsafe {
                fused_features_kernel_persist::launch_unchecked::<R>(
                    &client,
                    CubeCount::Static(cube_x, n_strips, 3),
                    CubeDim::new_3d(TX, 1, 1),
                    ArrayArg::from_raw_parts(src_a.clone(), pt),
                    ArrayArg::from_raw_parts(dst_a.clone(), pt),
                    ArrayArg::from_raw_parts(src_b.clone(), pt),
                    ArrayArg::from_raw_parts(dst_b.clone(), pt),
                    ArrayArg::from_raw_parts(src_c.clone(), pt),
                    ArrayArg::from_raw_parts(dst_c.clone(), pt),
                    ArrayArg::from_raw_parts(partials_f64.clone(), n_partials_f64),
                    ArrayArg::from_raw_parts(partials_max.clone(), n_partials_max),
                    ArrayArg::from_raw_parts(p_mu1.clone(), plane_len),
                    ArrayArg::from_raw_parts(p_mu2.clone(), plane_len),
                    ArrayArg::from_raw_parts(p_ssq.clone(), plane_len),
                    ArrayArg::from_raw_parts(p_s12.clone(), plane_len),
                    pw as u32,
                    ph as u32,
                    n_strips,
                    0u32,
                    0u32,
                    pt as u32,
                    0u32,
                    ph as u32,
                );
            }
        }

        // CPU reference planes + compare to the GPU-produced planes.
        let (cmu1, cmu2, cssq, cs12) = cpu_feature_planes(&src, &dst, pw, ph);
        let gmu1 = f32::from_bytes(&client.read_one(p_mu1.clone()).expect("read mu1")).to_vec();
        let gmu2 = f32::from_bytes(&client.read_one(p_mu2.clone()).expect("read mu2")).to_vec();
        let gssq = f32::from_bytes(&client.read_one(p_ssq.clone()).expect("read ssq")).to_vec();
        let gs12 = f32::from_bytes(&client.read_one(p_s12.clone()).expect("read s12")).to_vec();
        for (pi, (cpu_plane, gpu_plane)) in [
            (&cmu1, &gmu1),
            (&cmu2, &gmu2),
            (&cssq, &gssq),
            (&cs12, &gs12),
        ]
        .iter()
        .enumerate()
        {
            for off in 0..plane_len {
                let e = (cpu_plane[off] - gpu_plane[off]).abs();
                if e > 1e-3 {
                    plane_div_total += 1;
                }
                if e > worst_plane_err {
                    worst_plane_err = e;
                    let ch = off / pt;
                    let local = off % pt;
                    worst_plane_where = (s, ch * 4 + pi, local % pw, local / pw);
                }
            }
        }

        // Step 2 — per_scale reads the GPU-produced planes.
        let scale_dm = client.empty(pt * core::mem::size_of::<f32>());
        unsafe {
            per_scale_weighted_ssim_kernel::launch_unchecked::<R>(
                &client,
                cube_count(pt),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(p_mu1.clone(), plane_len),
                ArrayArg::from_raw_parts(p_mu2.clone(), plane_len),
                ArrayArg::from_raw_parts(p_ssq.clone(), plane_len),
                ArrayArg::from_raw_parts(p_s12.clone(), plane_len),
                ArrayArg::from_raw_parts(scale_dm.clone(), pt),
                pw as u32,
                ph as u32,
                pt as u32,
                w[0],
                w[1],
                w[2],
            );
        }
        // Step 3 — upsample-add into acc.
        unsafe {
            pow2x_upsample_add_kernel::launch_unchecked::<R>(
                &client,
                cube_count(base_n),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(scale_dm.clone(), pt),
                ArrayArg::from_raw_parts(acc.clone(), base_n),
                pw as u32,
                ph as u32,
                base_pw as u32,
                height as u32,
                s as u32,
                blend,
            );
        }

        // CPU mirror of the full chain.
        let dm_cpu = cpu_per_scale(&cmu1, &cmu2, &cssq, &cs12, pt, pt, w);
        cpu_upsample_add(&dm_cpu, pw, &mut acc_cpu, base_pw, height, s as u32, blend);
    }

    let gpu = f32::from_bytes(&client.read_one(acc.clone()).expect("read acc")).to_vec();
    let mut max_err = 0.0f32;
    let mut argmax = 0usize;
    let mut n_div = 0usize;
    for i in 0..base_n {
        let e = (gpu[i] - acc_cpu[i]).abs();
        if e > 1e-3 {
            n_div += 1;
        }
        if e > max_err {
            max_err = e;
            argmax = i;
        }
    }
    let plane_pass = worst_plane_err <= 1e-3;
    let acc_pass = max_err <= 1e-3;
    let (ws, wp, wx, wy) = worst_plane_where;
    let mode = mode.label();
    println!(
        "  [{mode}] {label} ({width}x{height}): \
         PLANES worst_err={worst_plane_err:.6} ({plane_div_total} slots>1e-3) \
         @scale{ws} plane{wp} (x={wx},y={wy}) -> {}  ||  \
         ACC max_err={max_err:.6} ({n_div}/{base_n}>1e-3) argmax(x={},y={}) gpu={} cpu={} -> {}",
        if plane_pass { "PASS" } else { "FAIL" },
        argmax % base_pw,
        argmax / base_pw,
        gpu[argmax],
        acc_cpu[argmax],
        if acc_pass { "PASS" } else { "FAIL" },
    );
    if !acc_pass {
        let mut shown = 0;
        for i in 0..base_n {
            if (gpu[i] - acc_cpu[i]).abs() > 1e-3 {
                println!(
                    "      acc (x={}, y={}) gpu={:.6} cpu={:.6}",
                    i % base_pw,
                    i / base_pw,
                    gpu[i],
                    acc_cpu[i]
                );
                shown += 1;
                if shown >= 8 {
                    break;
                }
            }
        }
    }
    plane_pass && acc_pass
}

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type Backend = cubecl::cpu::CpuRuntime;

fn main() {
    // Init a `log` backend so cubecl-wgpu's compile-failure errors surface.
    // Default to `error` so a silent Metal shader-compile failure is visible;
    // override with RUST_LOG (e.g. `cubecl_wgpu=trace`).
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("error")).init();
    let zero_fill = std::env::var("ZERO_FILL").map(|v| v != "0").unwrap_or(true);
    println!(
        "metal-diffmap-repro (zenmetrics#20) — backend={}, ZERO_FILL={}",
        std::any::type_name::<Backend>(),
        zero_fill
    );
    let mut all_pass = true;
    // 64x64 is the control (immune in the real pipeline); 96x80, 128x128 fail on Metal.
    // For each size we try every producer: host-upload (CI-confirmed PASS) plus the
    // on-device relays that exercise the producer→consumer storage read-after-write.
    let producers = [Producer::Host, Producer::Relay1d, Producer::Relay3d];
    for producer in producers {
        println!("── producer = {} ──", producer.label());
        for &(w, h) in &[(64usize, 64usize), (96, 80), (128, 128)] {
            all_pass &= run::<Backend>(&format!("{w}x{h}"), w, h, zero_fill, producer);
        }
    }

    // df64 precision probe — PER-KERNEL fast-math control. Two identical df64
    // two-sum kernels, differing only by `#[cube(precise)]`. On Metal the precise
    // one gets MTLCompileOptions fast-math OFF (recovers ~4096); the non-precise
    // one keeps Metal's default fast-math ON (collapses to ~0). Same machine, same
    // backend in one run — proves the per-kernel control. base=2^24, add 1.0 x4096.
    {
        let client = Backend::client(&Default::default());
        let run_probe = |precise: bool| -> [f32; 2] {
            let out = client.empty(2 * core::mem::size_of::<f32>());
            unsafe {
                if precise {
                    df64_probe_kernel::launch_unchecked::<Backend>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(1),
                        ArrayArg::from_raw_parts(out.clone(), 2),
                        16777216.0f32,
                        1.0f32,
                        4096u32,
                    );
                } else {
                    df64_probe_fast_kernel::launch_unchecked::<Backend>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(1),
                        ArrayArg::from_raw_parts(out.clone(), 2),
                        16777216.0f32,
                        1.0f32,
                        4096u32,
                    );
                }
            }
            let bytes = client.read_one(out.clone()).expect("read df64 probe");
            let r = f32::from_bytes(&bytes);
            [r[0], r[1]]
        };
        let pr = run_probe(true);
        let fa = run_probe(false);
        let verdict = |df: f32| {
            if (df - 4096.0).abs() < 1.0 {
                "WORKS"
            } else {
                "COLLAPSED"
            }
        };
        println!(
            "── df64 per-kernel fast-math probe (expect df64 recovered ~4096):\n\
                  precise   (#[cube(precise)], fast-math OFF on Metal): df64 = {:.1} -> {}\n\
                  non-precise (fast-math ON):                           df64 = {:.1} -> {}",
            pr[1],
            verdict(pr[1]),
            fa[1],
            verdict(fa[1]),
        );
    }

    // The faithful producer: the real feature kernel writes the persist planes.
    println!("── producer = feature-kernel (fused_features_kernel_persist) ──");
    for &(w, h) in &[(64usize, 64usize), (96, 80), (128, 128)] {
        all_pass &= run_feature::<Backend>(&format!("{w}x{h}"), w, h, FeatMode::Full);
    }

    // Minimized producer: blur-only (no f64, no partials, no SSIM math). PASSES on Metal.
    println!("── producer = blur-only (shared-mem sliding-window blur, no f64/features) ──");
    for &(w, h) in &[(64usize, 64usize), (96, 80), (128, 128)] {
        all_pass &= run_feature::<Backend>(&format!("{w}x{h}"), w, h, FeatMode::BlurOnly);
    }

    // The f64 isolator: blur-only + ONE f64 accumulator + f64 buffer write.
    println!("── producer = blur+f64 (blur-only + a single f64 accumulator) ──");
    for &(w, h) in &[(64usize, 64usize), (96, 80), (128, 128)] {
        all_pass &= run_feature::<Backend>(&format!("{w}x{h}"), w, h, FeatMode::BlurF64);
    }

    if all_pass {
        println!("ALL PASS (backend matches the CPU reference)");
    } else {
        println!("DIVERGENCE DETECTED (see FAIL rows above) — this is zenmetrics#20");
        std::process::exit(1);
    }
}
