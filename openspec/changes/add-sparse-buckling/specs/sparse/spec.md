# sparse Specification

## ADDED Requirements

### Requirement: Generalized shift-invert eigensolver for indefinite pencils

`scipp::sparse` SHALL provide `eigsh_gen(A, B, k, sigma, tol, maxiter)`, a
generalized shift-invert Lanczos solver for the symmetric pencil `A x = θ B x`
with `A` **symmetric (indefinite permitted)** and `B` **symmetric-positive-
definite**. It SHALL return the `k` eigenpairs whose eigenvalues θ are nearest
`sigma`, as an `EigshResult` carrying eigenvalues in **ascending** order,
`B`-normalized eigenvectors (`xᵀ B x = 1`, one per column), the total Lanczos
steps performed, and a `converged` flag. The shift-invert operator
`(A − σ B)⁻¹` SHALL be produced by a **single** sparse direct factorization
reused across the `B`-orthonormal Lanczos iterations — SPD Cholesky when
`(A − σ B)` is definite, otherwise LU — and SHALL NOT refactor per iteration nor
densify `A` or `B`. This is the shared primitive underneath `eigsh` (the SPD-`A`,
`sigma = 0` lowest-modes case) and `eigsh_buckling`. Convergence SHALL be gated on
the true generalized residual `‖A x − θ B x‖ / ‖A x‖`. (oracle: scipy/linalg —
`eigh(A, B)`)

#### Scenario: Nearest-sigma eigenpairs of an indefinite pencil
- GIVEN a symmetric-indefinite `A` and SPD `B`, and a shift `sigma` off the
  spectrum
- WHEN `eigsh_gen(A, B, k, sigma)` is computed
- THEN it returns `k` ascending eigenvalues θ nearest `sigma` matching a dense
  `eigh(A, B)` reference, each eigenvector satisfying
  `‖A x − θ B x‖ / ‖A x‖ < tol` and `xᵀ B x ≈ 1`, and `converged = true`

#### Scenario: SPD `eigsh` is preserved as a special case
- GIVEN an SPD pencil `K x = λ M x`
- WHEN `eigsh(K, M, k, sigma = 0)` is computed
- THEN it returns the same lowest-`k` ascending mass-normalized eigenpairs as
  before this change (behavior unchanged; `eigsh` delegates to the shared core)

### Requirement: Sparse buckling eigensolver (smallest positive load factors)

`scipp::sparse` SHALL provide `eigsh_buckling(K, K_geo, k, sigma0, tol, maxiter)`,
solving the linear-buckling generalized eigenproblem `(K + λ K_geo) φ = 0` with
`K` (elastic stiffness) **SPD** and `K_geo` (geometric stiffness) **symmetric,
typically indefinite**. It SHALL return a `BucklingResult` carrying the `k`
**smallest positive** load factors λ in **ascending** order (`load_factors`),
their `K`-normalized mode shapes (`φᵀ K φ = 1`, one per column of `modes`), the
Lanczos steps of the solve (`iterations`), the number of factorizations spent
locating the shift (`shifts`), and a `converged` flag. Non-positive load factors
SHALL be filtered out.

The solver SHALL reduce the problem to the pencil `K_geo φ = μ (K φ)` with
`μ = −1/λ` — so the smallest positive λ is the **algebraically most-negative** μ,
NOT the μ nearest `sigma = 0` — and SHALL locate a shift **strictly below the
whole spectrum** via an adaptive-σ walk driven by cheap factorization-only
definiteness probes (`(K_geo − σ K)` is SPD ⇔ σ is below the spectrum), then run a
**single** generalized shift-invert Lanczos solve at that shift and map the k
most-negative μ back to `λ = −1/μ` ascending. `sigma0 > 0` seeds the walk with a
trial load factor (`≤ 0` auto-scales from `‖K_geo‖_F / ‖K‖_F`); `maxiter ≤ 0`
picks the default. Convergence SHALL be gated on the true buckling residual
`‖K φ + λ K_geo φ‖ / ‖K φ‖`, and `converged` SHALL be `false` when fewer than `k`
positive load factors are resolved to `tol`. Results SHALL match a dense
`scipy.linalg.eigh(K_geo, K)` reference (filtered to positive `λ = −1/μ`,
ascending) within documented tolerance. (oracle: scipy/linalg — `eigh(K_geo, K)`)

#### Scenario: Smallest positive load factor of an indefinite buckling pencil
- GIVEN a buckling pencil with SPD `K` and indefinite `K_geo` whose pencil
  eigenvalues μ are mixed-sign (e.g. `μ = {−0.261, −0.102, +0.178}`, so the
  positive load factors are `λ = {3.837, 9.766}`)
- WHEN `eigsh_buckling(K, K_geo, k = 1)` is computed
- THEN it returns the smallest positive `λ ≈ 3.837` (the most-negative μ), NOT the
  nearest-`σ = 0` value `9.766`, and it does NOT return the negative
  factor `−1/0.178`

#### Scenario: Closed-form Euler beam element
- GIVEN the pinned-pinned single Euler-Bernoulli beam element reduced to its two
  end-rotation DOFs, `K = [[4, 2], [2, 4]]`, `K_geo = −(1/30)[[4, −1], [−1, 4]]`
- WHEN `eigsh_buckling(K, K_geo, k = 2)` is computed
- THEN the load factors are `λ ≈ {12, 60}` (the analytical discrete buckling
  loads), ascending, each mode satisfying `‖K φ + λ K_geo φ‖ / ‖K φ‖ < tol` and
  `φᵀ K φ ≈ 1`, and `converged = true`

#### Scenario: Every returned load factor is positive and ascending
- GIVEN any `eigsh_buckling` result with `converged = true`
- WHEN the `load_factors` are inspected
- THEN all are strictly positive and sorted ascending, and each corresponding
  mode satisfies the buckling residual and `K`-normalization checks

#### Scenario: Too few positive modes is signaled, not hidden
- GIVEN a pencil that has fewer than `k` positive load factors (or a budget too
  small to resolve them to `tol`)
- WHEN `eigsh_buckling(K, K_geo, k)` is computed
- THEN it returns `converged = false` with only the positive factors it resolved,
  rather than returning silently-wrong or negative load factors