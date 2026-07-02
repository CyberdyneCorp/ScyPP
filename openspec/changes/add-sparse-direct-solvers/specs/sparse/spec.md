## MODIFIED Requirements

### Requirement: Sparse linear solvers

`scipp::sparse` SHALL provide `spsolve` (direct), `cg` and `gmres` (matrix-free
iterative), and `norm`, matching SciPy within documented tolerance. `spsolve`
SHALL factor genuinely sparsely — it SHALL NOT densify the matrix (no
`toarray()` on the operating path) except as an explicit small-N fallback below a
documented size threshold; for `N` above that threshold it SHALL use a sparse
direct factorization (SPD → Cholesky/LDLᵀ, otherwise LU) with a fill-reducing
ordering. `cg` and `gmres` SHALL support a preconditioner and SHALL NOT return a
silently non-converged result: on reaching `maxiter` without meeting `tol` they
SHALL signal non-convergence via the reporting entry point. (oracle:
scipy/sparse/linalg)

#### Scenario: Direct and iterative solves
- GIVEN a sparse system `A x = b`
- WHEN `spsolve(A, b)`, `cg(A, b)` (SPD `A`) and `gmres(A, b)` are computed
- THEN each returns `x` with `A @ x` ≈ `b`, matching SciPy

#### Scenario: spsolve does not densify
- GIVEN an SPD sparse system of `N` DOF above the small-N threshold
- WHEN `spsolve(A, b)` is called
- THEN it produces `x` with `A @ x` ≈ `b` (matching SciPy) using a sparse
  factorization whose peak memory is O(nnz(L)) rather than O(N²)

#### Scenario: Non-convergence is signaled, not hidden
- GIVEN an iterative solve that reaches `maxiter` without meeting `tol`
- WHEN the reporting entry point (`cg`/`gmres` with an `IterationResult`) is used
- THEN the result reports `converged = false` with the achieved iteration count
  and final relative residual, rather than returning a silently-wrong vector

## ADDED Requirements

### Requirement: Preconditioned iterative solvers

`scipp::sparse::linalg` SHALL provide preconditioner selection for `cg` and
`gmres` — at minimum Jacobi/diagonal and `IC(0)` (incomplete Cholesky) for
SPD-CG, and `ILU(0)` for GMRES — and SHALL provide a reporting entry point that
returns an `IterationResult` carrying the solution, the number of iterations, the
final relative residual, and a `converged` flag. On a well-conditioned SPD system
the preconditioned solve SHALL converge in no more iterations than the
unpreconditioned solve. (oracle: scipy/sparse/linalg)

#### Scenario: Preconditioner accelerates convergence
- GIVEN a stiff/ill-conditioned SPD system where unpreconditioned CG does not
  converge within `maxiter`
- WHEN CG runs with an `IC(0)` (or Jacobi) preconditioner
- THEN it converges within `maxiter` to `A @ x` ≈ `b`, and the reported final
  residual is below `tol`

#### Scenario: Iteration result is reported
- GIVEN any preconditioned iterative solve
- WHEN the reporting entry point is used
- THEN it returns the solution together with `iterations`, `final_residual`, and
  `converged`, all consistent with the returned vector

### Requirement: Sparse direct factorization

`scipp::sparse::linalg` SHALL provide sparse direct factorization: Cholesky /
LDLᵀ for symmetric-positive-definite matrices and LU (Gilbert–Peierls,
left-looking, partial pivoting) for general matrices, each with a fill-reducing
ordering (RCM by default; `natural` and `amd` selectable, with `amd` permitted to
be added later). The factorization SHALL solve `A x = b` and SHALL match SciPy's
`spsolve` within documented tolerance. Cholesky SHALL report a failure (rather
than produce a wrong result) when the matrix is not positive definite. (oracle:
scipy/sparse/linalg)

#### Scenario: SPD Cholesky matches SciPy
- GIVEN an SPD sparse matrix `A` and RHS `b`
- WHEN the sparse Cholesky/LDLᵀ factorization solves `A x = b`
- THEN `x` is `allclose` to SciPy's `spsolve(A, b)` and `A @ x` ≈ `b`

#### Scenario: General LU matches SciPy
- GIVEN a general (non-symmetric) sparse matrix `A` and RHS `b`
- WHEN the sparse LU factorization solves `A x = b`
- THEN `x` is `allclose` to SciPy's `spsolve(A, b)`

#### Scenario: Fill-reducing ordering reduces fill
- GIVEN a sparse matrix with a fill-inducing natural ordering
- WHEN factored with the RCM ordering
- THEN the factor has no more nonzeros than the natural-ordering factor, and the
  solve still matches SciPy

#### Scenario: Non-SPD is rejected by Cholesky
- GIVEN a symmetric indefinite (or non-positive-definite) matrix
- WHEN a Cholesky factorization is attempted
- THEN it reports failure so the caller can fall back to LU, rather than
  returning a wrong solution

### Requirement: Reusable sparse factorization objects

`scipp::sparse::linalg` SHALL provide reusable factorization objects — `splu`
(sparse LU) and `factorized` (a solve functor), plus `spilu` (incomplete LU) for
preconditioning — that factor a matrix once and solve for many right-hand sides,
matching SciPy. Reusing a factorization for multiple RHS SHALL produce the same
result as an independent `spsolve` for each RHS. (oracle: scipy/sparse/linalg)

#### Scenario: Factor once, solve many
- GIVEN a sparse matrix `A` factored via `splu(A)` (or `factorized(A)`)
- WHEN the factorization solves `A x = b_k` for several right-hand sides `b_k`
- THEN each `x_k` matches SciPy's `spsolve(A, b_k)` without re-factorizing

### Requirement: Sparse solver diagnostics

`scipp::sparse` SHALL expose the reordering choice and matrix statistics (number
of nonzeros, bandwidth, and structural symmetry) used by the direct solver, so
users can tune ordering and reproduce results.

#### Scenario: Matrix statistics are reported
- GIVEN a sparse matrix
- WHEN its statistics are queried
- THEN `nnz`, bandwidth, and a symmetry indicator are reported and consistent
  with the matrix contents
