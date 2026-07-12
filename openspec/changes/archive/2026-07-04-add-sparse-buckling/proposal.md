## Why

`scipp::sparse` gained a shift-invert Lanczos generalized symmetric eigensolver
`eigsh(K, M, k, sigma)` in #12/#15 for `K x = λ M x` with **`M` SPD** — the
scalable path to the lowest modes of large sparse FE systems. Linear **buckling**
(eigenvalue) analysis needs a different pencil:

**`(K + λ K_geo) φ = 0`** — `K` the elastic stiffness (SPD), `K_geo` the
geometric/stress stiffness (**symmetric-indefinite**) — solved for the **k
smallest positive load factors λ** and their mode shapes. This is `*BUCKLE` in
CalculiX / Abaqus. The existing `eigsh` cannot express it: its target is
"eigenvalues nearest `sigma`", and the crux of buckling is that the smallest
positive λ corresponds to the **most-negative** pencil eigenvalue, not the one
nearest `sigma = 0`. A naive `sigma = 0` call returns the *largest* load factor
and can emit spurious negative (tension) factors that must be filtered.

Found while building **CalculiX++** (issue
[#18](https://github.com/CyberdyneCorp/SciPP/issues/18)): its Phase-4
`eigensolution` engine has only a **dense** reduction (`chol(K)` → standard
symmetric → `eigh`, then filter to positive λ ascending) with **no scalable path
beyond ~1500 free DOF**. It needs a sparse smallest-positive-λ solver, ideally
one that also exposes the underlying indefinite-`A`/SPD-`B` primitive.

## What Changes

Two layered, **additive** public entry points, both built on the existing
thick-restart Lanczos engine (`TRLanczos`) with **no new Krylov code**:

- **Low-level primitive (issue option b)** —
  `eigsh_gen(A, B, k, sigma, tol, maxiter)`: the generalized shift-invert Lanczos
  solver for `A x = θ B x` with **`A` symmetric (indefinite permitted)** and
  **`B` SPD**, returning the `k` eigenpairs nearest `sigma`, `B`-normalized,
  eigenvalues ascending. This is today's `eigsh` body generalized (the recurrence
  only ever required the inner-product matrix `B` to be SPD; `A`'s indefiniteness
  only affects the one-time factorization of `(A − σB)`, which already falls from
  Cholesky to LU). `eigsh(K, M, …)` keeps its exact signature/behavior and
  becomes a thin forwarder.

- **High-level buckling driver (issue option a)** —
  `eigsh_buckling(K, K_geo, k, sigma0, tol, maxiter)` returning a
  `BucklingResult{load_factors, modes, iterations, shifts, converged}`: the `k`
  **smallest positive** load factors λ **ascending** of `(K + λ K_geo) φ = 0`,
  with `K`-normalized modes (`φᵀ K φ = 1`). It maps the problem to the pencil
  `K_geo φ = μ (K φ)`, `μ = −1/λ`, and runs an **adaptive-σ walk** that places the
  shift **strictly below the whole spectrum** (where the k nearest-σ eigenvalues
  are exactly the k most-negative μ = k smallest positive λ), then does **one**
  `eigsh_gen` solve. The walk uses cheap **factorization-only definiteness
  probes**: since `B = K` is SPD, `(K_geo − σK) ≻ 0 ⇔ σ` lies below the spectrum,
  a Sturm bit the sparse factorizer already records as `used_cholesky`.

- **Detail accessor** — `detail::factorization_definite(F)` (one line over the
  already-stored `used_cholesky`/`ok`) so the walk reads the Sturm bit without
  any new inertia machinery.

## Capabilities

### Modified Capabilities
- `sparse`: the "Sparse generalized symmetric eigensolver" surface gains a
  reusable indefinite-`A`/SPD-`B` generalized primitive (`eigsh_gen`) and a
  smallest-positive-load-factor **buckling** driver (`eigsh_buckling`) alongside
  the existing `eigsh`.

## Impact

- **Code**: refactor the `eigsh` body in `src/sparse/eigen.cpp` into a shared
  `eigsh_core(A, B, …)` (behavior-preserving; `eigsh` + `eigsh_gen` both delegate,
  covered by existing `sp_eig_*`/`sp_clus_*` goldens); add `eigsh_buckling` + its
  small helpers in the same TU (reusing `Operator`/`select_nearest`/`gen_residual`
  verbatim); add `eigsh_gen`, `BucklingResult`, `eigsh_buckling` to
  `include/scipp/sparse/sparse.hpp`; add `factorization_definite` to
  `include/scipp/sparse/detail.hpp` + `src/sparse/factor.cpp`.
- **APIs**: additive — every existing signature (`eigsh`, `EigshResult`)
  preserved; `EigshResult` is **not** modified.
- **Tests**: two committed buckling pencils added to `tests/oracle/generate.py`
  and frozen into `tests/golden/golden.hpp` — a **closed-form pinned-pinned Euler
  beam element** (λ = {12, 60}) and a **discriminating indefinite** 3-DOF pencil
  (λ = {3.837, 9.766}) that fails loudly if the sign/target is inverted; oracle is
  `scipy.linalg.eigh(K_geo, K)` → `λ = −1/μ` filtered positive, ascending. New
  `tests/test_sparse.cpp` case checks load factors vs SciPy, the buckling residual
  `‖K φ + λ K_geo φ‖ / ‖K φ‖ < tol`, `φᵀ K φ ≈ 1`, and that **no** returned load
  factor is `≤ 0`.
- **Docs**: `README.md`, `CHANGELOG.md`, sparse spec updated.
- **Dependencies**: none (NumPP already pinned 1.6.0). **Version**: 1.4.0 → 1.5.0.