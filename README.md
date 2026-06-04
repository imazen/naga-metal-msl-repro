# naga Metal (MSL) miscompile — standalone repro + iterate harness

A compute shader that is **correct on Vulkan and CUDA** returns wrong results on
**Apple Metal**: a scattered subset of output elements come back holding a
**fixed value independent of the shader's inputs** — as if a dynamically-indexed
read from a `storage` buffer returns stale memory instead of the written value.
It only triggers once buffers exceed a size threshold (64×64 ok, 96×80+ fail).

The WGSL is plain (a `%`, two `>>`, an `if/else` min-clamp, one dynamic
`storage` read — see [`generated_wgsl.txt`](./generated_wgsl.txt)) and correct
everywhere except Metal, so the defect is in **naga's WGSL→MSL lowering** (or
Apple's MSL compiler). Originally found in
[imazen/zenmetrics#20](https://github.com/imazen/zenmetrics/issues/20).

## Layout

```
src/main.rs          the repro (cubecl): zero -> per_scale -> upsample chain over a
                     4-level pyramid, GPU vs a plain-Rust CPU reference; exits 1 on divergence
generated_wgsl.txt   the exact cubecl-emitted WGSL for the 3 kernels
UPSTREAM_REPORT.md    draft gfx-rs/wgpu issue (fill the [FILL ON METAL] blanks)
vendor/wgpu          submodule: imazen/wgpu fork @ v29.0.3, branch naga-metal-msl-repro
                     -> Cargo `[patch.crates-io] naga = { path = "vendor/wgpu/naga" }`
.github/workflows/ci.yml   `metal` job (macos-latest, real Metal) + `vulkan` control
```

## Run

```bash
git clone --recurse-submodules https://github.com/imazen/naga-metal-msl-repro
cd naga-metal-msl-repro
# macOS -> Metal -> EXPECTED FAIL at 96x80 / 128x128:
cargo run --release --no-default-features --features wgpu
# Linux/NVIDIA -> Vulkan, or --features cuda -> PASS (proves the WGSL is fine)
cargo run --release --no-default-features --features wgpu
```

`ZERO_FILL=0` toggles a defensive buffer zero-fill (already shown to make **no**
difference on Metal — the producing kernel writes every slot, so the wrong value
is *computed/read* from in-bounds-but-wrong data, not uninitialized memory).

Dump the WGSL naga is fed: `CUBECL_DEBUG_LOG=stdout cargo run ...`.

## The iterate loop (for a worker on GitHub Actions)

1. **Confirm**: push — the `metal` job fails, `vulkan` passes. That's the bug.
2. **Minimize**: shrink `src/main.rs` (drop sizes / kernels) and/or the WGSL to
   the smallest input that still fails the `metal` job. The upsample kernel
   (`pow2x_upsample_add_kernel`) is the prime suspect — the NN-replicate scatters
   one bad read into many outputs. If it's clean in isolation, suspect the
   channel-offset read (`idx + pad_total*{1,2}`) in `per_scale_weighted_ssim_kernel`.
3. **Fix**: patch the MSL backend in `vendor/wgpu/naga/src/back/msl/`. Commit in
   the submodule, bump the submodule pointer here, push. The `metal` job rebuilds
   the patched naga on a real Metal device and re-runs the repro — green when the
   miscompile is gone.
4. **Upstream**: once minimal + fixed, complete `UPSTREAM_REPORT.md` and open a
   PR / issue at `gfx-rs/wgpu`.

## What's already ruled out
- Not the algorithm / not the WGSL: identical WGSL is correct on Vulkan + CUDA.
- Not f32 precision: the wrong value is a *fixed* number, byte-identical across
  inputs that should change it — a wrong *read*, not arithmetic drift.
- Not uninitialized memory: zero-filling the buffer first changed nothing on
  Metal (the kernel writes every slot).
- Size-dependent and NN-replicated → one mistranslated dynamic `storage` index,
  not global corruption.
