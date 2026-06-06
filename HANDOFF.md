# HANDOFF — how to track down the upstream trigger

> ## ✅ RESOLVED (2026-06-05) — the trigger is `f64`, see [README.md](./README.md)
>
> The playbook below led to the answer. The upstream producer
> `fused_features_kernel_persist` was ported into `src/main.rs` and bisected on
> Metal CI: stripping it to a blur-only kernel **PASSES**, and adding back a
> single `f64` accumulator **FAILS**. Apple Metal has no `f64`; cubecl emits it
> anyway; wgpu rejects the module (`Using f64 values requires … FLOAT64`); the
> launch no-ops; the persist planes are left uninitialized → the `~1.098`
> garbage. **naga is innocent** (its validator correctly refuses `f64`). The fix
> is `f32` partials in zensim-gpu `fused.rs` (and/or cubecl downgrading/erroring
> on `f64` when the device lacks `SHADER_F64`). The historical playbook follows.

---


This repo's isolated 3-kernel chain (`src/main.rs`) does **not** reproduce on
Metal (CI-confirmed PASS at 64×64 / 96×80 / 128×128). So `per_scale_weighted_ssim`
and `pow2x_upsample_add` are correctly translated on Metal **with clean,
host-uploaded inputs**. The real bug (a fixed `~1.098` value at scattered pixels,
sizes ≥96, on Apple Metal only) therefore comes from **upstream** — either the
GPU kernels that *produce* the persist planes, or a producer→consumer storage
hazard naga mistranslates only when those buffers are written by a prior dispatch.

Source of truth for the bug + all measurements: **imazen/zenmetrics#20**.

## The single decisive experiment to run first

zensim-gpu already has a debugging accessor:
`crates/zensim-gpu/src/pipeline.rs:807` — `debug_read_persist_plane(scale, plane_idx)`
reads back a persist plane (`mu1`/`mu2`/`ssq`/`s12`) for a scale.

On a **Metal** machine, run the failing test
(`crates/zensim-gpu/tests/cpu_gpu_diffmap_parity.rs`) for fixture 1 (96×80) and,
at the diffmap argmax pixel (it reported e.g. `(x=2, y=44)`, gpu=`1.0981547`),
read the persist planes at the corresponding coarse-scale coordinate via
`debug_read_persist_plane`. Compare to the CPU reference's blurred moments.

- **If the persist-plane value is already garbage on Metal** → the bug is in the
  feature/blur kernel (`fused::fused_features_kernel_persist`), or its XYB /
  downscale inputs. The diffmap kernels are innocent (this repo already proved
  that). Port THAT kernel into `src/main.rs` next.
- **If the persist planes are correct but the diffmap output is still wrong** →
  it's a producer→consumer hazard: the same buffer is correct when host-uploaded
  (this repo) but wrong when written by a prior GPU dispatch. Add a trivial GPU
  "producer" kernel here that writes the persist planes on-device, then run
  `per_scale` — if that fails on Metal, you've minimized it.

This one read-back tells you which half of the pipeline to chase, without
guessing.

## The upstream kernels, in order (the suspects), with source locations

All in `imazen/zenmetrics : crates/zensim-gpu/src/` (the diffmap path runs these
to fill `persist_planes_ref` BEFORE `run_gpu_diffmap_chain` at `pipeline.rs:2941`):

1. `kernels/color.rs` — `srgb_to_positive_xyb_kernel` (launched `pipeline.rs:1552, 1789`): sRGB→XYB opsin.
2. `kernels/downscale.rs` — `downscale_2x_3ch_kernel` (`pipeline.rs:1576, 1816`): pyramid downscale.
3. `kernels/blit.rs` — `copy_rows_kernel` (`pipeline.rs:1661`).
4. **`kernels/fused.rs:489` — `fused_features_kernel_persist`** (launched `pipeline.rs:1917`, via `launch_blur_and_features_persist` at `pipeline.rs:1895`): the V-blur + SSIM-moment kernel that WRITES `mu1/mu2/ssq/s12` (`persist_planes_ref`, `pipeline.rs:375`). **Prime suspect** — it's the producer the diffmap consumes.

The diffmap consumer chain (already ruled out here) is in `kernels/diffmap.rs`
and `run_gpu_diffmap_chain` (`pipeline.rs:2941`).

## Two concrete paths to a reproducer

**Path A — port the feature kernel into this repo (keeps it standalone).**
Copy `fused_features_kernel_persist` (+ a minimal XYB/downscale producer for its
inputs) into `src/main.rs`, feed it synthetic XYB, run it → `per_scale` →
`upsample`, diff vs CPU. Push; the `metal` CI job tells you if it now fails.
Bisect: add the producers one at a time. This yields a minimal upstream repro.

**Path B — run the real zensim test on Metal (guaranteed reproducer, heavier).**
The actual failing test reproduces today. To run it with the patched naga:
```bash
# in a checkout of imazen/zenmetrics (clone the sibling path-deps exactly as
# .github/workflows/ci.yml "Clone sibling-repo path-dep targets" pins them —
# notably ../../zensim @ 1b3eaa3d), add to its workspace Cargo.toml:
#   [patch.crates-io] naga = { path = "<this repo>/vendor/wgpu/naga" }
cargo test -p zensim-gpu --no-default-features --features wgpu --release \
  --test cpu_gpu_diffmap_parity gpu_diffmap_matches_cpu_canonical_pointwise -- --nocapture
# It SOFT-asserts: prints a DIFFMAP DIVERGENCE report per fixture. Patch
# vendor/wgpu/naga/src/back/msl until those reports go away.
```
Minimize from there by trimming the test to one fixture/scale, then deleting
kernels until it stops failing.

## Key data (from #20, all on Apple-Silicon Metal; CUDA + Vulkan are correct)

- Fixed garbage value per size, **identical across all distortion levels**:
  96×80 → `1.0981547` @ (2,44); 128×128 → `1.0981432` @ (69,127);
  160×120 → `0.1518212`; 200×160 → `0.05175012`. CPU ≈ `0.0001` there.
- **64×64 is immune** (its pyramid has no coarse planes with this footprint).
- A constant value independent of the distortion ⇒ not arithmetic; a wrong/stale
  *read*. Zero-filling the diffmap scratch made **zero** difference (the kernel
  writes every slot) ⇒ not uninitialized memory; it's computed/read wrong.
- The aggregate **score** path averages the same persist planes and stays correct
  — consistent with a *few* bad per-pixel values that wash out in a sum.

## Tooling
- Dump the WGSL naga is fed: `CUBECL_DEBUG_LOG=stdout cargo run ...` (this repo
  already saved it to `generated_wgsl.txt` for the 3 consumer kernels; do the
  same to capture the feature-kernel WGSL once you've added it).
- Capture the MSL naga emits on Metal (naga snapshot / `Device::create_shader_module`
  trace) and diff WGSL→MSL for the offending kernel.
- The naga MSL backend you patch: `vendor/wgpu/naga/src/back/msl/`.

## Don't
- Don't post `UPSTREAM_REPORT.md` to gfx-rs/wgpu until a **standalone** repro
  actually fails on Metal (the draft says so). The current isolated kernels pass,
  so an upstream report built only on them would be wrong.
