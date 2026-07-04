## Context

FE modal/frequency/buckling analysis needs the lowest-N eigenpairs of the sparse
generalized symmetric problem `K x = λ M x` (`K` symmetric, `M` SPD). The dense
reduction (`chol(M)` → standard symmetric → `eigh`) is O(n³) and caps at a few
thousand DOF; only the lowest modes are wanted, so an iterative Krylov method on
the sparse operator is the scalable path. SciPP already has (from #10) a sparse
direct factorization — the exact building block a shift-invert scheme needs.

## Goals / Non-Goals

- **Goals**: lowest-`k` (or nearest-`sigma`) eigenpairs of a sparse SPD pencil;
  reuse the #10 factorization (factor once, apply many); mass-normalized vectors,
  ascending eigenvalues, a `converged` flag; match SciPy's `eigsh` within
  tolerance.
- **Non-Goals**: non-symmetric / complex `eigs`; ARPACK-style implicit restarts
  (full reorthogonalization is used instead — simpler and robust for the modest
  subspace sizes modal analysis needs); a preconditioned `lobpcg`.

## Decisions

### Shift-invert Lanczos in the M-inner product
The operator `C = (K − σ M)⁻¹ M` is self-adjoint with respect to
`⟨a,b⟩_M = aᵀ M b`. Running Lanczos in that inner product yields a symmetric
tridiagonal `T`; its eigenvalues `θ` relate to the pencil's by
`θ = 1/(λ − σ)`, so the eigenvalues **nearest `σ`** are the **largest |θ|** —
exactly what Lanczos converges to first. Shift back as `λ = σ + 1/θ`. For an SPD
pencil `sigma = 0` gives the lowest modes; a small negative shift steps off a
rigid-body/near-zero mode (where `K` itself is singular).

### Factor once, reuse
`(K − σ M)` is factored a single time into an opaque `Factorization` (SPD →
sparse Cholesky, else sparse LU, both from #10) and applied at every Lanczos
step via `factorization_solve`. This is the whole efficiency argument: each
Lanczos iteration is one sparse triangular back/forward solve, not a refactor.
The same builder now backs `spsolve`/`factor_nnz`, removing duplicated
SPD-vs-LU selection logic.

### Full reorthogonalization + true-residual convergence
The M-orthonormal basis is fully reorthogonalized (twice) each step, which keeps
`V` M-orthonormal and suppresses the spurious/ghost eigenvalues that plague
plain Lanczos — affordable for the small subspaces (`ncv ≈ max(2k+1, 20)`) modal
analysis uses. Convergence is gated on the **true** generalized residual
`‖K x − λ M x‖ / ‖K x‖ < tol` for each returned mode, not just the tridiagonal
residual estimate, so a reported `converged = true` is trustworthy. The Krylov
dimension grows up to the subspace cap until the `k` wanted modes converge.

### Projected problem on NumPP dense eigh
The small `m × m` tridiagonal is solved with `numpp::linalg::eigh` (NumPP 1.6.0
tred2/tql2, O(m³)). Ritz vectors `V s` are M-orthonormal by construction, hence
already mass-normalized (`xᵀ M x = 1`).

## Risks / Trade-offs

- **`(K − σ M)` singular** (e.g. `σ` exactly an eigenvalue, or `σ = 0` with
  rigid-body modes): the factorization fails; `eigsh` returns `converged = false`
  with empty results. Mitigation: document choosing `σ` off the spectrum / off
  zero. Not silently wrong.
- **No implicit restart**: memory is O(n · ncv). Fine for the lowest tens of
  modes; a restarted variant is deferred.

## Migration

Additive. NumPP pin moves 1.5.0 → 1.6.0 (already the local bootstrap version).
</content>
