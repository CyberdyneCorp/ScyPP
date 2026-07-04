## MODIFIED Requirements

### Requirement: Sparse generalized symmetric eigensolver

`scipp::sparse` SHALL provide `eigsh(K, M, k, sigma, tol, maxiter)`, a
shift-invert Lanczos solver for the generalized symmetric eigenproblem
`K x = λ M x` with `K` symmetric and `M` symmetric-positive-definite. It SHALL
return the `k` eigenpairs whose eigenvalues are nearest `sigma` (the lowest `k`
modes for an SPD pencil at `sigma = 0`), as an `EigshResult` carrying
eigenvalues in **ascending** order, **mass-normalized** eigenvectors
(`xᵀ M x = 1`, one per column), the total Lanczos steps performed, and a
`converged` flag. The shift-invert operator `(K − σ M)⁻¹` SHALL be produced by a
single sparse direct factorization reused across the Lanczos iterations (it
SHALL NOT refactor per iteration and SHALL NOT densify `K` or `M`).

The solver SHALL be robust to operator scale and to clustered spectra: it SHALL
declare an invariant subspace only **relative** to the operator scale (never on
an absolute `beta` threshold), and SHALL **restart** (thick/implicit restart,
deflating converged Ritz pairs) rather than stop when the wanted modes have not
converged within one Krylov cycle. `maxiter` bounds the number of restart cycles
(`≤ 0` selects a default). Convergence SHALL be gated on the true generalized
residual `‖K x − λ M x‖ / ‖K x‖`. Results SHALL match SciPy's
`scipy.sparse.linalg.eigsh` within documented tolerance. (oracle:
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

#### Scenario: Stiff pencil with large-magnitude eigenvalues
- GIVEN an SPD pencil whose eigenvalues are large (e.g. `λ ~ 1e10`), so the
  shift-invert operator's eigenvalues (and the Lanczos `beta`) are `~1e-11`
- WHEN `eigsh(K, M, k, sigma = 0)` is computed
- THEN the small `beta` is NOT mistaken for an invariant subspace, and the `k`
  modes converge to `‖K x − λ M x‖ / ‖K x‖ < tol` matching SciPy

#### Scenario: Clustered spectrum converges via restart
- GIVEN an SPD pencil with tightly-clustered / near-degenerate low modes that a
  single Krylov cycle cannot resolve to `tol`
- WHEN `eigsh(K, M, k, sigma = 0)` is computed
- THEN it restarts across cycles and converges the `k` modes to `tol`, matching
  SciPy, with `converged = true`

#### Scenario: Non-convergence is signaled, not hidden
- GIVEN a restart-cycle budget too small to resolve the `k` requested modes to
  `tol`
- WHEN `eigsh` exhausts `maxiter` cycles without meeting `tol`
- THEN the result reports `converged = false` (with the Lanczos steps
  performed), rather than returning silently-wrong eigenpairs
</content>
