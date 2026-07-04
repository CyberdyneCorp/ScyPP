## Context

`eigsh` (#12) runs a single-shot M-orthonormal Lanczos on the shift-invert
operator `C = (K − σ M)⁻¹ M`, growing the Krylov dimension to a cap. Two
assumptions break on real FE pencils: that `beta` reaching `1e-12` means an
invariant subspace (false when the operator itself is `~1e-11`), and that a large
single subspace resolves any spectrum (false for clustered modes).

## Decisions

### Relative breakdown threshold
The invariant-subspace test becomes `beta ≤ ε · opscale`, where `opscale` is a
running max of the `|alpha|` and `beta` magnitudes seen so far (`ε = 1e-12`). A
genuine invariant subspace gives `beta/opscale ~ machine-eps`; the stiff pencil's
legitimate small `beta` is `~1e-2` of `opscale` and no longer trips it. This is
the standard ARPACK-style relative test.

### Internal operator rescaling
Scale the operator by `s = trace(K)/trace(M)` (the Rayleigh-quotient scale of the
pencil): `Ĉ = s · C`, so `Ĉ`'s eigenvalues `θ = s/(λ − σ)` are O(1) and the
tridiagonal/arrowhead entries stay well-scaled for the dense `eigh`. The
shift-invert factor `K − σ M` is unchanged (`s` only multiplies the operator
output); shift back with `λ = σ + s/θ`. `s` is guarded to a finite positive value
(else `1`). This makes the relative-threshold choice scale-insensitive and
removes the caller-side pencil rescaling.

### Thick-restart Lanczos (Wu & Simon)
When the `k` wanted Ritz pairs have not converged and the subspace cap `ncv` is
reached, restart instead of stopping. TRLan maintains the relation
`Ĉ V = V H + β_last v_{ncv+1} eᵀ`; for a Ritz pair `(θ_i, y_i = V s_i)` this gives
`Ĉ y_i = θ_i y_i + (β_last s_i[last]) v_{ncv+1}`. So retaining `nkeep` Ritz
vectors nearest `σ` yields an **arrowhead** projected matrix — diagonal of their
Ritz values, a spike `β_last s_i[last]` coupling each to the residual vector
`v_{ncv+1}` — and Lanczos continues from that residual with a three-term
recurrence. Converged pairs are effectively deflated; the rest keep refining.
`nkeep = min(ncv−1, k + (ncv−k)/2)`. Full reorthogonalization (twice, in the
M-inner product) keeps the basis M-orthonormal across restarts.

The first cycle (`nkeep = 0`, no spike) is exactly the #12 single-shot Lanczos,
so well-separated spectra converge in one cycle unchanged.

### API reinterpretation (source-compatible)
`maxiter` becomes the max number of restart cycles (`≤ 0` → a default of 200);
`iterations` reports the total Lanczos steps performed across cycles. The
`eigsh`/`EigshResult` signatures are unchanged. Convergence stays gated on the
true generalized residual, so `converged = true` is trustworthy.

## Risks / Trade-offs

- **Exact multiplicities**: single-vector Lanczos (even thick-restarted) resolves
  one eigenvector per exactly-degenerate eigenvalue; genuine multiplicities would
  need block Lanczos. Near-degenerate (clustered) spectra — the FE case — are
  handled. Documented; block Lanczos deferred.
- **`(K − σ M)` singular** (σ on the spectrum, or σ = 0 with rigid-body modes):
  unchanged from #12 — the factorization fails and `eigsh` returns
  `converged = false`. Choosing σ off the spectrum is still required.

## Migration

Additive/behavioral only; no signature or dependency change. Downstream callers
can drop any pre-scaling of the pencil.
</content>
