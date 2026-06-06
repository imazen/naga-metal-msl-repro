# zenmetrics#20 root cause — `f64` in a compute kernel breaks on Apple Metal

> ## ✅ SOLVED (2026-06-05): the trigger is `f64`, not a naga MSL miscompile
>
> The Metal failure is **not** a WGSL→MSL lowering bug. It is `f64`:
>
> 1. Apple-Silicon Metal does **not** support 64-bit floats — the wgpu adapter
>    advertises no `SHADER_F64` (confirmed in CI: the device feature list has
>    `SHADER_F16`, `SHADER_INT64`, … but no `SHADER_F64`).
> 2. cubecl emits `f64` WGSL for the feature kernel anyway (it does **not**
>    downgrade to `f32` when the device lacks the feature).
> 3. wgpu **correctly rejects** the module at `create_shader_module`:
>    `Shader validation error: Using f64 values requires the
>    naga::valid::Capabilities::FLOAT64 flag`.
> 4. cubecl logs that error (via `log`) but the kernel launch then no-ops, so the
>    persist planes are left **uninitialized** → downstream `per_scale`/`upsample`
>    read garbage → the scattered fixed `~1.098` in the full pipeline.
>
> **naga's MSL backend never runs** — validation fails first, and its validator is
> behaving *correctly* (Metal has no `double`). So patching `vendor/wgpu/naga`
> cannot fix this. The fix belongs upstream of naga (see **The fix** below).

## How it was reproduced and minimized (all on real Apple-Silicon Metal CI)

The repro (`src/main.rs`) drives the real zensim-gpu diffmap producer/consumer
kernels under several "producers" and diffs GPU output against a plain-Rust CPU
reference. CUDA (RTX 5070) and Vulkan/llvmpipe pass every mode; only Metal fails.

| producer of the mu1/mu2/ssq/s12 planes | Metal | note |
|---|---|---|
| `host-upload` (planes uploaded clean) | **PASS** | consumers `per_scale`+`upsample` are fine |
| `gpu-relay-1d` / `gpu-relay-3d` (planes copied on-device) | **PASS** | a plain producer→consumer read-after-write is fine |
| `fused_features_kernel_persist` (the real producer, **uses f64** partials) | **FAIL** | planes come back garbage |
| `blur-only` (the producer **stripped of f64**/partials/feature math) | **PASS** | the shared-mem sliding-window blur is fine |
| `blur+f64` (`blur-only` **+ one `f64` accumulator + one `Array<f64>` write**) | **FAIL** | the isolated trigger |

`blur-only` and `blur+f64` differ by exactly one `f64` — that one addition flips
Metal from correct to garbage (and corrupts even the kernel's unrelated `f32`
plane writes, because the whole module fails validation and never runs).

## The fix

Apple GPUs have no `f64`. The kernel must not use it on the wgpu/Metal path.

- **zensim-gpu (the shipping fix):** make the `partials_f64` accumulators in
  `crates/zensim-gpu/src/kernels/fused.rs` (`fused_features_kernel_persist` and
  the sibling `fused_features_kernel`) `f32` on the wgpu/Metal path. The persist
  planes are already `f32`, so the **diffmap output is unaffected by the
  precision change**; only the score-path partials lose a little precision, which
  the metric tolerates. This makes `fused_features_kernel_persist` validate and
  run on Metal.
- **cubecl (the robustness fix):** when the wgpu adapter lacks `SHADER_F64`,
  either downgrade `f64`→`f32` in the WGSL compiler or **hard-error at kernel
  build** instead of emitting an `f64` module that fails validation and then
  silently produces uninitialized-buffer garbage. The silent-garbage failure
  mode is what made this bug so hard to find.
- **naga / `vendor/wgpu`:** nothing to fix — the validator correctly refuses
  `f64` for a Metal target.

## Layout

```
src/main.rs          the repro (cubecl): producers (host/relay/feature/blur/blur+f64)
                     -> per_scale -> upsample over a 4-level pyramid, GPU vs a
                     plain-Rust CPU reference; reads back persist planes too; exit 1 on divergence
generated_wgsl.txt   cubecl-emitted WGSL for the 3 consumer kernels
vendor/wgpu          submodule: imazen/wgpu fork @ v29.0.3 (innocent; kept for the
                     [patch.crates-io] naga override the original investigation used)
.github/workflows/ci.yml   `metal` job (macos-latest, real Metal) + `vulkan` control;
                     the metal "Capture …" step uploads metal_run.log (WGSL + RUST_LOG
                     diagnostics) as an artifact — that log is where the validation error is
```

## Run

```bash
git clone --recurse-submodules https://github.com/imazen/naga-metal-msl-repro
cd naga-metal-msl-repro
# macOS -> Metal: feature/blur+f64 modes FAIL (f64 unsupported); the rest PASS.
cargo run --release --no-default-features --features wgpu
# Linux/NVIDIA -> Vulkan (or --features cuda) -> ALL PASS (those backends have f64).
cargo run --release --no-default-features --features cuda
```

Set `RUST_LOG=cubecl_wgpu=trace` to see the `WGSL compilation failed … Using f64
values requires … FLOAT64` validation error that is the smoking gun.

## What was ruled out along the way
- Not the consumer kernels (`per_scale`, `upsample`): correct on Metal with both
  host-uploaded and GPU-relayed planes.
- Not a producer→consumer storage hazard: trivial on-device relays pass on Metal.
- Not the blur / shared memory / mirror indexing / sliding window: `blur-only` passes.
- Not naga's WGSL→MSL lowering: naga never runs — wgpu validation rejects `f64` first.
- **It is `f64`**, which Apple Metal does not support and cubecl emits regardless.
