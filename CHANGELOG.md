# Changelog

## 1.2.0 — 2026-07-02 — sparse direct solvers and preconditioned iterative solvers

Fixes [#10](https://github.com/CyberdyneCorp/SciPP/issues/10): `scipp::sparse`
gains genuinely-sparse direct factorization and preconditioned iterative solvers,
so finite-element stiffness systems (SPD) beyond a few hundred DOF are now
viable. Built on **NumPP 1.5.0** and validated against **SciPy 1.15** as the
numerical oracle: **142 cases / 7603 checks, 0 divergences** (clang, gcc, and
ASan/UBSan). Phases 1–2 of the `add-sparse-direct-solvers` OpenSpec change;
reusable factorization objects and diagnostics (Phases 3–4) are tracked for
follow-up.

### Sparse direct factorization — `spsolve` no longer densifies
- `spsolve` previously called `numpp::linalg::solve(A.toarray(), b)` — O(N²)
  memory / O(N³) time (≈0.5 GB / 95 s for an 8268-DOF FE system). It now factors
  genuinely sparsely (peak memory O(nnz(L))), in `src/sparse/factor.cpp`:
  - **Cholesky** (up-looking, elimination-tree/`ereach`) for symmetric
    positive-definite systems — the FE stiffness case.
  - **LU** (Gilbert–Peierls left-looking, partial pivoting) for general systems.
  - `spsolve` routes SPD → Cholesky, otherwise (or on a non-positive pivot) → LU,
    keeping a dense fallback only for very small `N` (≤ 64).
- **Fill-reducing ordering** — reverse Cuthill–McKee (`OrderingMethod::Rcm`,
  the default) with `Natural`; `Amd` is reserved and currently resolves to RCM.
  On an arrow matrix RCM cuts factor fill from 465 to 59 nonzeros.
- `factor_nnz(A, ordering)` reports factor fill for tuning; `spsolve(A, b,
  ordering)` always takes the sparse path.

### Preconditioned iterative solvers + convergence reporting
- `cg`/`gmres` accept a `Preconditioner` — `None`, `Jacobi` (inverse diagonal),
  `IC0` (zero-fill incomplete Cholesky, for SPD-CG), and `ILU0` (zero-fill
  incomplete LU, for GMRES); a failed incomplete factorization degrades to Jacobi
  (`src/sparse/precond.cpp`).
- New `cg_report`/`gmres_report` return
  `IterationResult { x, iterations, final_residual, converged }` and **signal
  non-convergence** rather than returning a silently-wrong vector at `maxiter` —
  closing the correctness hazard where unpreconditioned CG hit `maxiter` ~16 %
  off with no warning. GMRES is right-preconditioned so the Krylov residual
  equals the true residual. The existing `cg`/`gmres`/`spsolve` signatures are
  preserved.

## 1.1.0 — 2026-06-28 — special functions, `odr`, GPU acceleration, `just`, and the SciPP rename

The first published release. Builds on the v1.0.0 12-phase foundation with a large
additive feature pass, real GPU offload, a task-runner workflow, and the project
rename. Built on **NumPP 1.5.0** and validated against **SciPy 1.15** as the
numerical oracle: **140 cases / 7503 checks, 0 divergences**.

### Project rename — `ScyPP` → `SciPP`
- The project, namespace (`scypp::` → `scipp::`), include tree (`include/scipp/`),
  CMake project/targets, backend flags (`SCIPP_WITH_*`), and docs were renamed to
  **SciPP**, matching the GitHub repository `CyberdyneCorp/SciPP`. Pure mechanical
  change, no behaviour difference.

### GPU acceleration (now real, via NumPP 1.5.0)
- The deferred device-kernel paths are wired to NumPP's new acceleration
  primitives — each auto-selects a device backend above the size threshold and
  always falls back to the portable CPU path, reporting the choice via
  `last_backend()`:
  - **sparse** — `spmv`/`spmm` (CSR) → `numpp::csr_spmv`.
  - **spatial** — `cdist`/`pdist` euclidean/sqeuclidean → `numpp::cdist_euclidean`.
  - **ndimage** — `correlate1d` and the gaussian/uniform/derivative separable
    passes → `numpp::correlate1d`.
- Pins the NumPP dependency floor to **1.5.0**.

### `scipp::special` — the deferred function tail
- **Airy**: `airy`, `airye`.
- **Elliptic**: `ellipk`, `ellipkm1`, `ellipe`, `ellipkinc`, `ellipeinc`, `ellipj`.
- **Error-function relatives**: `erfcx`, `dawsn`, `wofz`/`voigt_profile`, `fresnel`.
- **Spherical Bessel**: `spherical_jn`/`yn`/`in`/`kn`; **integrals** `sici`, `shichi`.
- **Misc**: `lambertw`, `zeta`/`zetac`, `struve`/`modstruve`, `spence`.
- **Hypergeometric**: `hyp0f1`, `hyp1f1`, `hyp2f1`, `hyperu`.

### New `scipp::odr` capability
- Orthogonal distance regression (`Model`/`Data`/`ODR`) — total-least-squares fit
  with estimated parameters, standard errors, and residual variance.

### Capability drawdowns
- **stats** — discrete distributions (`binom`/`poisson`/`geom`/`bernoulli`/`nbinom`/
  `hypergeom`), nonparametric rank tests (`mannwhitneyu`/`wilcoxon`/`kruskal`/
  `kendalltau`), and normality tests (`shapiro`/`anderson`).
- **optimize** — `linprog` (two-phase simplex), `nnls` (Lawson–Hanson), and the
  `Powell`/`CG`/`L-BFGS-B` `minimize` methods (L-BFGS-B with box bounds).
- **integrate** — stiff solvers `solve_ivp` `Radau` and `BDF`, `solve_bvp`
  (collocation), and nested/extended quadrature (`romberg`, `quad_vec`, `dblquad`,
  `tplquad`, `nquad`).
- **interpolate** — `griddata` (nearest/linear) and B-splines (`BSpline`,
  `make_interp_spline`, `splev`).
- **fft** — DCT/DST `ortho`/`forward` norms (types I–IV) and N-D `dctn`/`idctn`/
  `dstn`/`idstn`.
- **signal** — discrete-time LTI (`cont2discrete`, `dstep`/`dimpulse`/`dlsim`,
  `dbode`/`dfreqresp`).
- **sparse** — `csgraph` traversals and flow (`breadth_first_order`,
  `depth_first_order`, `johnson`, `maximum_flow`, `maximum_bipartite_matching`).

### Tooling & process
- **`justfile`** task runner (`bootstrap`/`build`/`test`/`ctest`/`debug`/`gcc`/
  `asan`/`oracle`/`spec`/`ci`/`clean`), mirroring NumPP/SymPP.
- **Phase 0 foundation closed out**: added `scipp::linalg_error`, a foundation
  regression test, and a CI workflow (`openspec validate` + CPU build/test).
- **Deferred-work tracking**: every open backlog is mirrored as a labeled GitHub
  issue (`deferred`/`openspec`/`module:*`), cross-linked from its OpenSpec change;
  the convention (and the bug → regression-test rule) is documented in
  `openspec/project.md`.

## 1.0.0 — 2026-06-27 — initial 12-phase SciPy parity

First clean-room C++20 port of SciPy's commonly-used surface, built on NumPP and
validated against SciPy as the oracle. Ships all 12 phases:

- **special** + **constants**, **linalg**, **fft**/**fftpack**, **optimize**,
  **integrate** + **differentiate**, **interpolate**, **stats**, **signal**,
  **sparse**, **spatial**, **ndimage**, **cluster** + **io**.
- Foundation: CMake/C++20 project, pinned NumPP `find_package` dependency, the
  `scipp::error` model, the tiered-acceleration dispatch shim over NumPP's
  `CapabilityRegistry`/`last_backend()`, and the frozen-golden SciPy oracle
  harness for Python-free CI.
