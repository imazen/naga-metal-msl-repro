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

// ───────────────────────── harness ─────────────────────────

const N_SCALES: usize = 4;

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

fn run<R: Runtime>(label: &str, width: usize, height: usize, zero_fill: bool) -> bool {
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

        let mu1_h = client.create_from_slice(f32::as_bytes(&mu1));
        let mu2_h = client.create_from_slice(f32::as_bytes(&mu2));
        let ssq_h = client.create_from_slice(f32::as_bytes(&ssq));
        let s12_h = client.create_from_slice(f32::as_bytes(&s12));

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
        "  {label} ({width}x{height}, base_pw={base_pw}): max_err = {max_err:.6} \
         ({n_div}/{base_n} px > 1e-3); argmax (x={}, y={}) gpu={} cpu={}  -> {}",
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

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type Backend = cubecl::cpu::CpuRuntime;

fn main() {
    let zero_fill = std::env::var("ZERO_FILL").map(|v| v != "0").unwrap_or(true);
    println!(
        "metal-diffmap-repro (zenmetrics#20) — backend={}, ZERO_FILL={}",
        std::any::type_name::<Backend>(),
        zero_fill
    );
    let mut all_pass = true;
    // 64x64 is the control (immune in the real pipeline); 96x80, 128x128 fail on Metal.
    for &(w, h) in &[(64usize, 64usize), (96, 80), (128, 128)] {
        all_pass &= run::<Backend>(&format!("{w}x{h}"), w, h, zero_fill);
    }
    if all_pass {
        println!("ALL PASS (backend matches the CPU reference)");
    } else {
        println!("DIVERGENCE DETECTED (see FAIL rows above) — this is zenmetrics#20");
        std::process::exit(1);
    }
}
