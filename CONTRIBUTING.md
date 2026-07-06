# Contributing to SciPP

Thanks for your interest in SciPP — a modern C++20 port of SciPy, built on
[NumPP](https://github.com/CyberdyneCorp/NumPP). This guide covers how to build, the
workflow we follow, and what a mergeable change looks like.

## Getting set up

```bash
git clone https://github.com/CyberdyneCorp/SciPP.git
cd SciPP
just bootstrap      # build + install the pinned NumPP into .deps/  (expects ../NumPP)
just test           # configure + build + run the SciPy-oracle test suite (CPU backend)
```

No `just`? The equivalent is `scripts/bootstrap_numpp.sh` then `cmake -S . -B build` +
`cmake --build build` + `ctest --test-dir build`. See the [README](README.md#building).
GPU work: `just gpu-detect`, then `just configure -DSCIPP_WITH_CUDA=ON` (or `OPENCL`/`METAL`)
— a GPU backend requires a NumPP package built with the matching backend.

## Development workflow

We develop spec-first with [OpenSpec](https://github.com/Fission-AI/OpenSpec). Living
capability specs are in [`openspec/specs/`](openspec/specs); in-flight changes live under
[`openspec/changes/`](openspec/changes).

- **Medium or large features** (new SciPy subpackage surface, backends, build/packaging
  behavior): start with an OpenSpec change (proposal → specs/tasks) before implementing. Run
  `openspec validate --all --strict` — CI enforces it.
- **Small fixes and docs:** a direct PR is fine. Still keep the specs and docs truthful — if
  your change makes a spec or doc statement false, update it in the same PR.

## What a mergeable PR looks like

- **Tests.** New behavior ships with tests. **Every bug fix includes a regression test** that
  fails before the fix and passes after.
- **Green CI.** The CPU-only build + oracle suite (Linux, GCC) and OpenSpec validation must
  all pass.
- **Docs/specs in sync.** Update `README.md` and `openspec/specs/` when your change affects
  documented behavior. Don't leave a claim that the code contradicts.
- **Readable, low-complexity code.** Match the surrounding style (Concepts, Ranges, smart
  pointers, contiguous data — no raw owning pointers, no legacy macros). Keep per-function
  cognitive complexity modest; isolate genuinely irreducible numerical kernels and flag them
  rather than mangling them to hit a number.
- **A descriptive PR message.** Explain what changed and why. If you found a bug, describe how
  it reproduced.

## Commit & PR style

- Concise, technical, imperative commit subjects (e.g. "Fix branch cut in `special::sici`").
- Reference the OpenSpec change or issue when there is one.
- Keep unrelated changes in separate PRs.

## Numerical correctness

SciPP is a clean-room port validated against **real SciPy** as a numerical oracle. Golden
data is frozen under `tests/oracle/` so CI runs without Python; regenerate it with
`just oracle` (requires `python3` + `scipy`) after changing the test set. When you port or
change an algorithm, validate it against the SciPy reference within a documented tolerance and
add an oracle case if a gap exists. Spec requirements cite the SciPy source they port as
breadcrumbs, e.g. `(oracle: scipy/linalg/_decomp_lu.py)`.

## Reporting bugs & requesting features

Open an issue using the templates. For **security** issues, do **not** open a public issue —
see [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions are licensed under the project's
[MIT License](LICENSE).
