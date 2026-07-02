## Why

`scipp::sparse::spsolve` does not actually solve sparsely — it densifies the CSR
matrix (`A.toarray()`) and calls a dense LU, so it is O(N²) memory / O(N³) time
(an 8268-DOF FE system needs ~0.5 GB dense and ~95 s). `scipp::sparse::cg` is
genuinely sparse but **unpreconditioned**, so on realistically-conditioned FE
stiffness systems it hits `maxiter` without converging and returns a
silently-wrong vector (~16 % error observed on the same 8268-DOF system, with no
warning). Neither sparse-solver path is viable for finite-element stiffness
systems beyond a few hundred DOF (issue
[#10](https://github.com/CyberdyneCorp/SciPP/issues/10), found while building
CalculiX++ on top of `scipp::sparse`).

## What Changes

Full scope, delivered **phased across several PRs** (see `tasks.md` for the phase
boundaries). Each phase is independently shippable and testable against SciPy.

- **Phase 1 — Preconditioned iterative solvers + convergence reporting** (the
  correctness hazard):
  - Add preconditioners: Jacobi/diagonal and incomplete Cholesky `IC(0)` (SPD),
    `ILU(0)` for GMRES.
  - Make `cg`/`gmres` preconditioner-aware and expose an `IterationResult`
    (solution, iterations, final residual, `converged` flag). **BREAKING**-free:
    keep the existing `cg`/`gmres` signatures returning `ndarray`; add
    `cg_solve`/`gmres_solve` (or an options+info overload) that report
    convergence and **signal non-convergence** instead of returning a
    silently-wrong vector at `maxiter`.
- **Phase 2 — Real sparse direct factorization** (the missing piece the
  `spsolve` code comment flags):
  - Fill-reducing ordering: **RCM (reverse Cuthill-McKee) as the default**, with
    an ordering-selection enum (`natural`, `rcm`, and `amd` as a later addition).
  - Sparse **Cholesky / LDLᵀ** for the symmetric-positive-definite case (FE
    stiffness is SPD) — symbolic (elimination tree) + numeric factorization +
    triangular solves.
  - Sparse **LU** (Gilbert–Peierls left-looking, partial pivoting) for the
    general case.
  - Route `spsolve` to the sparse factorization (SPD → Cholesky, else LU),
    keeping the dense path only as a fallback for very small N.
- **Phase 3 — Reusable factorization objects**: `splu` / `factorized` — factor
  once, solve many RHS (needed for eigensolvers, multiple load cases, adjoint
  sensitivity). `spilu` (incomplete LU) for preconditioning.
- **Phase 4 — Diagnostics**: expose reordering choice and matrix statistics
  (nnz, bandwidth, symmetry) for tuning and reproducibility.

## Capabilities

### New Capabilities
<!-- none — this extends the existing sparse capability -->

### Modified Capabilities
- `sparse`: the "Sparse linear solvers" requirement changes — `spsolve` must
  factor genuinely sparsely (not densify) with a fill-reducing ordering; `cg`/
  `gmres` gain preconditioning, convergence reporting, and non-convergence
  signaling; new requirements add sparse direct factorization (Cholesky/LDLᵀ and
  LU), reusable factorization objects (`splu`/`factorized`/`spilu`), and matrix
  diagnostics.

## Impact

- **Code**: `src/sparse/linalg.cpp` (rewrite `spsolve`, extend `cg`/`gmres`),
  new `src/sparse/factor.cpp` (ordering, symbolic/numeric factorization,
  triangular solves) and `src/sparse/precond.cpp` (Jacobi/IC(0)/ILU(0)),
  `include/scipp/sparse/sparse.hpp` (new public types: `OrderingMethod`,
  `Preconditioner`, `IterationResult`, `SparseLU`/`SparseCholesky` /
  `factorized`), `src/sparse/detail.hpp` (CSC helpers).
- **APIs**: additive — existing `spsolve`/`cg`/`gmres`/`norm` signatures
  preserved; new solver entry points and factorization classes added.
- **Tests**: new SPD and general sparse systems in `tests/oracle/generate.py`
  (scipy `spsolve`, `splu`, `cg` with preconditioner) frozen into
  `tests/golden/golden.hpp`; new cases in `tests/test_sparse.cpp`. Existing
  golden values for the 4×4 `sp_A` system remain unchanged.
- **Dependencies**: none new — uses existing NumPP dense `solve` only as the
  small-N fallback. No METIS/AMD external libs (AMD, if added, is implemented
  in-tree in a later PR).
- **Backlog**: graduates `splu`/`spilu`/`factorized` out of the
  `add-sparse-extras` tracking change.
