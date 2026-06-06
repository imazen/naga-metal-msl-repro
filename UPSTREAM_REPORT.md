# Upstream report — cubecl emits `f64` on Metal (no `SHADER_F64`) → silent garbage

> **Target: cubecl (tracel-ai/cubecl; we run the `zenforks-cubecl` fork @ 0.10.1).**
> **NOT gfx-rs/wgpu / naga** — the earlier draft of this file blamed naga's MSL
> backend; that was wrong. naga's validator correctly refuses `f64` for Metal;
> it never gets the chance to mis-lower anything. Standalone repro:
> `imazen/naga-metal-msl-repro` (the `metal` CI job fails, `vulkan`/CUDA pass).

## Summary

A cubecl compute kernel that uses `f64` runs correctly on CUDA and Vulkan but
produces garbage on Apple-Silicon Metal. Apple GPUs do not support 64-bit floats
— the wgpu Metal adapter advertises no `SHADER_F64`. cubecl emits `f64` WGSL for
the kernel regardless of device support; wgpu then rejects the module at
`create_shader_module`:

```
Shader validation error: Type [2] '' is invalid
 = Using `f64` values requires the `naga::valid::Capabilities::FLOAT64` flag
```

cubecl surfaces this only via `log::error!` and the kernel launch then no-ops, so
the kernel's output buffers are left **uninitialized**. Downstream kernels read
that garbage. In our metric (zensim-gpu) this manifested as a scattered, fixed
`~1.098` in the per-pixel diffmap, sizes ≥96, identical across inputs — a classic
"reads stale memory" symptom that is actually "the producing kernel never ran."

## Environment

- cubecl: `zenforks-cubecl` 0.10.1 (tracks tracel-ai/cubecl 0.10.x).
- wgpu/naga: 29.0.3.
- Device: GitHub `macos-latest` Apple-Silicon runner ("Apple Paravirtual
  device", Metal backend, macOS 15). Adapter features include `SHADER_F16`,
  `SHADER_INT64`, `SUBGROUP`, … but **not** `SHADER_F64`.
- Correct backends: CUDA (RTX 5070) and Vulkan (llvmpipe + NVIDIA) — both have f64.

## Reproduction

`imazen/naga-metal-msl-repro`, `cargo run --release --no-default-features
--features wgpu` on macOS. The `metal` CI job runs it on a real device. The
minimization in `src/main.rs` isolates the trigger to a single `f64`:

- `blur-only` kernel (no f64) → **PASS** on Metal.
- `blur+f64` kernel = `blur-only` + one `f64` accumulator + one `Array<f64>`
  write → **FAIL** on Metal.

Run with `RUST_LOG=cubecl_wgpu=trace` (or read the `metal-wgsl` CI artifact) to
see the validation error above.

## Two issues, really

1. **cubecl emits `f64` for a device that doesn't support it.** It should either
   downgrade `f64`→`f32` in the WGSL compiler when the adapter lacks
   `SHADER_F64`, or refuse the kernel at build time with a clear error — not emit
   a module that fails validation.
2. **The failure is silent.** When `create_shader_module` returns a validation
   error, the launch should propagate a hard error, not no-op and leave output
   buffers uninitialized for the next kernel to read. The silent path turns a
   clear "unsupported feature" into a data-corruption bug that takes days to
   trace.

## Ask

Please confirm the intended behavior for `f64` kernels on devices without
`SHADER_F64`, and whether cubecl can (a) downgrade or (b) hard-error rather than
silently producing uninitialized output. Happy to test a patch against this
repro on real Metal CI.
