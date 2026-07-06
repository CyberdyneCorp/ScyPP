# Changelog

## 1.6.0 — 2026-07-06 — production-readiness: install/export, governance, consumability spec

Makes SciPP consumable by others without vendoring, and formalizes the "usable by
others" bar.

### Packaging
- **Install/export rules**: `find_package(SciPP CONFIG)` now works. Installs the
  `scipp` target and headers, exports `SciPPTargets` under the `scipp::` namespace,
  and generates `SciPPConfig`/`SciPPConfigVersion`. `SciPPConfig` re-resolves the
  transitive NumPP dependency via `find_dependency(NumPP CONFIG)`. The library carries
  `VERSION`/`SOVERSION` for a pinnable ABI. Verified end-to-end with a downstream
  `find_package(SciPP)` consumer that configures, links `scipp::scipp`, and runs.
- Synced the version across `CMakeLists.txt`, `conanfile.py`, and `vcpkg.json` (the
  Conan/vcpkg manifests had drifted to `1.0.0`).

### Governance & docs
- Added `CONTRIBUTING.md`, `SECURITY.md` (private vulnerability reporting),
  `CODE_OF_CONDUCT.md`, issue templates, and a pull-request template.
- Added the `consumability` OpenSpec capability spec — a portable readiness rubric.
- README: documented the installed-package consumption path and a
  **Versioning & API stability** section (semver + `SOVERSION`).

## 1.5.0 — 2026-07-04 — sparse buckling eigensolver: indefinite `A`, SPD `B`, smallest positive `λ`

Adds [#18](https://github.com/CyberdyneCorp/SciPP/issues/18): a scalable path to
the **linear-buckling** eigenproblem `(K + λ K_geo) φ = 0` (elastic stiffness `K`
SPD, geometric stiffness `K_geo` symmetric-**indefinite**), needed by CalculiX++
`*BUCKLE` which had only a dense `O(n³)` reduction (no path beyond ~1500 DOF).
Both additions layer on the existing thick-restart Lanczos engine — **no new
Krylov code**. Validated against a dense `scipy.linalg.eigh(K_geo, K)` oracle:
**146 cases / 7710 checks, 0 divergences** (clang, gcc, ASan).

### Generalized shift-invert primitive `eigsh_gen` (issue option b)
- `eigsh_gen(A, B, k, sigma, tol, maxiter)` solves the symmetric pencil
  `A x = θ B x` with `A` symmetric (**indefinite permitted**) and `B` SPD,
  returning the `k` eigenpairs nearest `sigma`, `B`-normalized, θ ascending. The
  existing `eigsh(K, M, …)` becomes a thin forwarder to the shared core; its
  signature and behavior are unchanged (covered by the existing `sp_eig_*` /
  `sp_clus_*` goldens).
- The operator scale is hardened to `s = max(|trace(A)|, trace(B))/trace(B)` so it
  stays well-conditioned when `A` has non-positive trace — identical to the old
  `trace(K)/trace(M)` for every SPD pencil (the scale cancels in `λ = σ + s/θ`).

### Buckling driver `eigsh_buckling` (issue option a)
- `eigsh_buckling(K, K_geo, k, sigma0, tol, maxiter)` returns a `BucklingResult
  { load_factors, modes, iterations, shifts, converged }` — the `k` **smallest
  positive** load factors `λ` ascending, with `K`-normalized modes (`φᵀ K φ = 1`).
- Maps the pencil to `K_geo φ = μ (K φ)`, `μ = −1/λ`, so the smallest positive `λ`
  is the algebraically **most-negative** `μ` — *not* the `μ` nearest `σ = 0` (a
  naive `σ = 0` target returns the largest factor and spurious negative factors).
- Locates the shift via an **adaptive-σ walk** using cheap factorization-only
  definiteness probes: because `B = K` is SPD, `(K_geo − σ K)` is SPD ⇔ `σ` lies
  below the whole spectrum — a free Sturm bit the sparse factorizer already records
  (`used_cholesky`), exposed as `detail::factorization_definite`. A geometric
  bracket + a few bisections place `σ*` below all modes, then a **single**
  `eigsh_gen` solve yields the wanted modes. Non-positive load factors are filtered;
  `converged` is `false` when fewer than `k` positive factors are resolved.

### Tests
- Two committed buckling pencils in `tests/oracle/generate.py` frozen into
  `tests/golden/golden.hpp`: a closed-form pinned-pinned Euler beam element
  (`λ = {12, 60}`) and a discriminating indefinite 3-DOF pencil (`λ = {3.837,
  9.766}`) that fails loudly if the sign/target is inverted. New `test_sparse.cpp`
  case checks load factors vs oracle, the buckling residual `‖K φ + λ K_geo φ‖ /
  ‖K φ‖`, `φᵀ K φ ≈ 1`, that **no** returned factor is `≤ 0`, the `eigsh_gen`
  primitive on the indefinite pencil, and the too-few-positive-modes signal.

## 1.4.0 — 2026-07-04 — robust `eigsh`: thick-restart Lanczos + relative breakdown

Fixes [#15](https://github.com/CyberdyneCorp/SciPP/issues/15): the `eigsh`
generalized eigensolver (#12) failed on **stiff** finite-element pencils and on
**tightly-clustered** spectra. Both traced to the single-shot Lanczos driver in
`src/sparse/eigen.cpp`. Validated against **SciPy 1.15**: **145 cases / 7673
checks, 0 divergences** (clang, gcc, ASan).

### Relative breakdown threshold + internal rescaling (bug #1)
- The invariant-subspace test was **absolute** (`beta > 1e-12`). For a stiff
  pencil (`E ≈ 2.1e5`, `ρ ≈ 7.8e-9`) the eigenvalues are `λ ≈ 1e10 … 1e13`, so
  the shift-invert operator's eigenvalues — and the Lanczos `beta` — legitimately
  sit at `~1e-11` and tripped the threshold after ~6 steps, spuriously capping
  the subspace (higher modes came out as noise). The test is now **relative** to
  a running operator scale.
- The shift-invert operator is internally **rescaled** by `s = trace(K)/trace(M)`
  so the recurrence runs at O(1) magnitude (`λ = σ + s/θ`), removing the need for
  the caller-side pencil rescaling downstream code used as a workaround.

### Thick-restart Lanczos (bug #2)
- When the wanted modes have not converged within one Krylov cycle, `eigsh` now
  **thick-restarts** (Wu & Simon): it retains the Ritz vectors nearest `σ` (their
  Ritz values become the projected diagonal, an arrowhead spike couples them to
  the residual) plus the residual vector, and continues — deflating converged
  pairs so **clustered / near-degenerate** FE spectra converge. Full
  reorthogonalization keeps the basis M-orthonormal across restarts.

### API note (source-compatible)
- No signature change to `eigsh` / `EigshResult`. `maxiter` is reinterpreted as
  the maximum number of **restart cycles** (`≤ 0` selects a default of 200), and
  `iterations` now reports the total Lanczos steps performed across cycles.
  Convergence remains gated on the true generalized residual
  `‖K x − λ M x‖ / ‖K x‖`.

## 1.3.1 — 2026-07-04 — `just gpu-detect` backend probe

Developer-tooling only — no library or API changes.

- Add a `just gpu-detect` recipe: a no-build, cross-platform (Linux / macOS /
  Windows) probe that reports which GPU backends are usable on the host
  (**CUDA** via `nvidia-smi`, **OpenCL** via `clinfo` / ICD loader, **Metal** on
  Apple) and recommends the matching `configure` flag
  (`-DSCIPP_WITH_{CUDA,OPENCL,METAL}=ON`), falling back to the always-available
  CPU backend. Since SciPP's GPU backends come from NumPP, the recommendation
  notes that the chosen backend also needs a NumPP built with the matching
  `NUMPP_WITH_*` flag.

## 1.3.0 — 2026-07-03 — sparse generalized symmetric eigensolver (eigsh)

Fixes [#12](https://github.com/CyberdyneCorp/SciPP/issues/12): `scipp::sparse`
gains a shift-invert Lanczos generalized symmetric eigensolver, the scalable
path to the lowest-N modes of large sparse SPD systems that FE modal / frequency
/ buckling analysis needs (the dense generalized eigensolve caps at a few
thousand DOF). Built on **NumPP 1.6.0** (O(n³) `eigh`, resolving NumPP#138) and
validated against **SciPy 1.15**: **143 cases / 7627 checks, 0 divergences**
(clang and ASan). Implements the `add-sparse-eigsh` OpenSpec change.

### Sparse generalized symmetric eigensolver — `eigsh`
- `eigsh(K, M, k, sigma = 0, tol = 1e-8, maxiter = 0)` solves the sparse pencil
  `K x = λ M x` (`K` symmetric, `M` SPD) for the `k` eigenpairs nearest `sigma`
  (the lowest modes at `sigma = 0`), in `src/sparse/eigen.cpp`:
  - **Shift-invert**: `(K − σ M)` is factored **once** (reusing the #10 sparse
    direct factorization) and `(K − σ M)⁻¹` is applied across the iterations —
    each Lanczos step is one sparse triangular solve, not a refactor.
  - **M-orthonormal Lanczos** on `C = (K − σ M)⁻¹ M` (self-adjoint in the
    `M`-inner product) with full reorthogonalization; the small projected
    tridiagonal is solved with NumPP's dense `eigh`; eigenvalues shift back as
    `λ = σ + 1/θ`.
  - Returns `EigshResult { eigenvalues, eigenvectors, iterations, converged }` —
    **ascending** eigenvalues, **mass-normalized** eigenvectors (`xᵀ M x = 1`),
    and a `converged` flag gated on the true generalized residual
    `‖K x − λ M x‖ / ‖K x‖`, so non-convergence is signaled rather than hidden.
- **Reusable sparse factorization** — `src/sparse/factor.cpp` refactored into a
  factor-once/solve-many `Factorization` (SPD → Cholesky, else LU), now shared by
  `spsolve` / `factor_nnz` / `eigsh` (no behavior change to the linear solvers).

### Dependency
- NumPP pin **1.5.0 → 1.6.0** for the O(n³) symmetric eigensolver.

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
