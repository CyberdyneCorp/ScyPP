> GitHub issue: [#10](https://github.com/CyberdyneCorp/SciPP/issues/10)
>
> Phased delivery — each numbered group is an independently shippable PR with its
> own SciPy oracle tests. Land Phase 1 first (silent non-convergence is a
> correctness hazard).

## 1. Phase 1 — Preconditioned iterative solvers + convergence reporting

- [x] 1.1 Add public types to `include/scipp/sparse/sparse.hpp`: `enum class Preconditioner { None, Jacobi, IC0, ILU0 }` and `struct IterationResult { ndarray x; int iterations; double final_residual; bool converged; }`
- [x] 1.2 Add `src/sparse/precond.cpp` (+ detail decls): Jacobi (inverse diagonal), `IC0` incomplete Cholesky and `ILU0` incomplete LU restricted to `A`'s sparsity pattern; each exposes an `apply(r) -> M⁻¹ r` via triangular solves
- [x] 1.3 Add reporting solver entry points (e.g. `cg_report` / `gmres_report`) that take a `Preconditioner`, run the preconditioned iteration, and return `IterationResult` (iterations, final relative residual, `converged`); signal non-convergence at `maxiter` instead of returning silently
- [x] 1.4 Refactor existing `cg`/`gmres` to delegate to the reporting core (unchanged signatures/behavior for `Preconditioner::None`)
- [x] 1.5 Oracle: add a stiff/ill-conditioned SPD system to `tests/oracle/generate.py` where unpreconditioned CG stalls but IC(0)/Jacobi CG converges; freeze golden data
- [x] 1.6 Tests in `tests/test_sparse.cpp`: preconditioned CG converges (`A@x ≈ b`), `IterationResult` fields consistent, non-convergence reports `converged=false`; run and pass
- [x] 1.7 Update `README.md` / sparse docs for the new preconditioner + reporting API

## 2. Phase 2 — Fill-reducing ordering + sparse direct factorization

- [x] 2.1 Add CSR↔CSC and permutation helpers to `include/scipp/sparse/detail.hpp` (transpose, symmetric permute `P A Pᵀ`, apply/invert permutation to a vector)
- [x] 2.2 Add `OrderingMethod { Natural, Rcm, Amd }` to the header; implement RCM (pseudo-peripheral start, degree-ordered BFS, reversed) in `src/sparse/factor.cpp`; `Amd` documented as delegating to RCM/Natural until its own PR
- [x] 2.3 Implement sparse triangular solves (lower/upper, CSC, DFS-reachability sparse-RHS variant) in `src/sparse/factor.cpp`
- [x] 2.4 Implement up-looking sparse Cholesky/LDLᵀ (elimination tree + column counts → symbolic; numeric factorization) with non-positive-pivot detection/failure signal
- [x] 2.5 Implement Gilbert–Peierls left-looking sparse LU with partial pivoting
- [x] 2.6 Route `spsolve`: small-N (≤ threshold) → dense `numpp::linalg::solve` fallback; else apply ordering, try Cholesky when structurally symmetric, fall back to LU; ensure the existing 4×4 `sp_A` oracle stays on the dense path (unchanged golden)
- [x] 2.7 Oracle: add a larger SPD FE-like system and a general non-symmetric system to `generate.py` (`spsolve` reference); freeze golden data
- [x] 2.8 Tests: SPD Cholesky and general LU solves match SciPy and `A@x ≈ b`; RCM factor nnz ≤ natural factor nnz; non-SPD rejected by Cholesky; run and pass
- [x] 2.9 Update docs for `spsolve` sparse behavior + ordering selection

## 3. Phase 3 — Reusable factorization objects

- [ ] 3.1 Add `SparseLU` / `SparseCholesky` classes with `.solve(b)` (and multi-RHS `.solve(B)`) reusing a stored factorization
- [ ] 3.2 Add free functions `splu(A, ordering)`, `factorized(A)` (solve functor), and `spilu(A)` (incomplete LU) to the header/impl
- [ ] 3.3 Oracle + tests: factor once, solve several RHS; each matches SciPy `spsolve(A, b_k)`; `spilu` usable as a GMRES preconditioner; run and pass
- [ ] 3.4 Update `add-sparse-extras` backlog: mark `splu`/`spilu`/`factorized` delivered; update docs

## 4. Phase 4 — Diagnostics

- [ ] 4.1 Add matrix-statistics API (nnz, bandwidth, structural-symmetry indicator) and expose the ordering actually used by the direct solver
- [ ] 4.2 Tests: statistics consistent with matrix contents; ordering choice observable; run and pass
- [ ] 4.3 Final docs pass (README sparse section, examples) and OpenSpec validate

## 5. Cross-cutting

- [x] 5.1 Keep per-function cognitive complexity within the systems band (≤ ~25–35); isolate factorization inner loops behind well-named helpers; measure with the cognitive-complexity skill
- [x] 5.2 Run full `tests/` suite and diff against `main`; ensure no regression
- [ ] 5.3 `openspec validate add-sparse-direct-solvers --strict` passes before archiving
