## Why

`scipp::sparse` has sparse direct + preconditioned iterative **linear** solvers
(from #10) but no **sparse eigensolver**. Finite-element modal / frequency /
buckling analysis needs the generalized symmetric eigenproblem
**`K x = λ M x`** for the lowest-N modes of large sparse SPD systems. The only
existing path is the **dense** generalized eigensolve via `numpp::linalg` —
capped at a few thousand DOF even now that NumPP's dense `eigh` is O(n³)
(NumPP#138) — so there is **no scalable path** to eigenvalues.

Found while building **CalculiX++** (C++20 FE solver, issue
[#12](https://github.com/CyberdyneCorp/SciPP/issues/12)): its Phase-4
`eigensolution` engine implements the dense reduction (Cholesky of `M` →
standard symmetric → `eigh`) and had to **defer** the scalable Lanczos because
SciPP exposed no sparse generalized eigensolver. A 1,275-DOF `*FREQUENCY` solve
takes ~111 s on the dense path.

## What Changes

- Add `scipp::sparse::eigsh(K, M, k, sigma, tol, maxiter)` — a shift-invert
  Lanczos generalized symmetric eigensolver returning the `k` eigenpairs nearest
  `sigma` (the lowest modes for an SPD pencil at `sigma = 0`).
  - **Shift-invert**: factor `(K − σ M)` **once** with the sparse direct
    factorization from #10 and apply `(K − σ M)⁻¹` across the Lanczos iterations
    (the shift-invert operator IS a sparse solve).
  - **M-orthonormal Lanczos**: run Lanczos in the `M`-inner product so the
    operator `C = (K − σ M)⁻¹ M` is self-adjoint; solve the small projected
    tridiagonal problem with NumPP's dense `eigh`; shift back `λ = σ + 1/θ`.
  - Return **mass-normalized** eigenvectors (`xᵀ M x = 1`) and **ascending**
    eigenvalues, plus an `EigshResult{eigenvalues, eigenvectors, iterations,
    converged}` — the `converged` flag mirrors `IterationResult::converged`.
- Expose a **reusable sparse factorization** in the `sparse` detail layer
  (`sparse_direct_factorize` / `factorization_solve`) so the shift-invert
  operator factors once and applies many times; `spsolve` / `factor_nnz` are
  rewritten on top of it with no behavior change.
- Bump the pinned **NumPP dependency to 1.6.0** for the O(n³) symmetric
  eigensolver used by the projected problem.

## Capabilities

### Modified Capabilities
- `sparse`: the "Sparse linear solvers" surface gains a generalized symmetric
  eigensolver (`eigsh`) alongside the existing linear solvers.

## Impact

- **Code**: new `src/sparse/eigen.cpp` (shift-invert Lanczos); refactor
  `src/sparse/factor.cpp` to a reusable `Factorization` (factor-once/solve-many)
  reused by `spsolve`/`factor_nnz`/`eigsh`; new public `EigshResult` + `eigsh`
  in `include/scipp/sparse/sparse.hpp`; detail decls in
  `include/scipp/sparse/detail.hpp`.
- **APIs**: additive — all existing signatures preserved.
- **Tests**: a generalized SPD pencil (`K = tri(-1,2,-1)`, consistent mass
  `M = tri(1,4,1)/6`, n=64) added to `tests/oracle/generate.py`
  (`scipy.sparse.linalg.eigsh(K, M, sigma=0)` reference) and frozen into
  `tests/golden/golden.hpp`; new case in `tests/test_sparse.cpp` checks
  eigenvalues vs SciPy, the generalized residual, mass-normalization, and
  M-orthogonality.
- **Dependencies**: NumPP pin **1.5.0 → 1.6.0** (dense O(n³) `eigh`). No new
  external libraries.
</content>
