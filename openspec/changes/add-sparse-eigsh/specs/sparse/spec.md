## ADDED Requirements

### Requirement: Sparse generalized symmetric eigensolver

`scipp::sparse` SHALL provide `eigsh(K, M, k, sigma, tol, maxiter)`, a
shift-invert Lanczos solver for the generalized symmetric eigenproblem
`K x = λ M x` with `K` symmetric and `M` symmetric-positive-definite. It SHALL
return the `k` eigenpairs whose eigenvalues are nearest `sigma` (the lowest `k`
modes for an SPD pencil at `sigma = 0`), as an `EigshResult` carrying
eigenvalues in **ascending** order, **mass-normalized** eigenvectors
(`xᵀ M x = 1`, one per column), the Lanczos subspace dimension built, and a
`converged` flag. The shift-invert operator `(K − σ M)⁻¹` SHALL be produced by a
single sparse direct factorization reused across the Lanczos iterations (it
SHALL NOT refactor per iteration and SHALL NOT densify `K` or `M`). Results SHALL
match SciPy's `scipy.sparse.linalg.eigsh` within documented tolerance. (oracle:
scipy/sparse/linalg)

#### Scenario: Lowest modes of an SPD pencil
- GIVEN a sparse SPD pencil `K x = λ M x` above the small-N threshold
- WHEN `eigsh(K, M, k, sigma = 0)` is computed
- THEN it returns `k` ascending eigenvalues matching SciPy, with each
  eigenvector satisfying `‖K x − λ M x‖ / ‖K x‖ < tol` and `xᵀ M x ≈ 1`, and
  `converged = true`

#### Scenario: Mass-orthonormal eigenvectors
- GIVEN the eigenpairs returned by `eigsh`
- WHEN two distinct modes `x_i`, `x_j` are compared
- THEN `x_iᵀ M x_j ≈ 0` and `x_iᵀ M x_i ≈ 1` (M-orthonormal, mass-normalized)

#### Scenario: Non-convergence is signaled, not hidden
- GIVEN a subspace cap too small to resolve the `k` requested modes to `tol`
- WHEN `eigsh` reaches the cap without meeting `tol`
- THEN the result reports `converged = false` (with the subspace dimension
  built), rather than returning silently-wrong eigenpairs
</content>
