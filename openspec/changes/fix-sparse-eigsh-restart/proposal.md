## Why

`scipp::sparse::eigsh` (v1.3.0, #12) works on well-conditioned pencils but fails
on **stiff finite-element pencils in consistent units** and on
**tightly-clustered spectra** (issue
[#15](https://github.com/CyberdyneCorp/SciPP/issues/15), found adopting `eigsh`
as CalculiX++'s `*FREQUENCY` eigensolver). Two root causes, both in the
single-shot Lanczos driver in `src/sparse/eigen.cpp`:

1. **Absolute breakdown threshold** — the invariant-subspace test
   `if (!(beta > 1e-12)) break;` is *absolute*. The shift-invert operator
   `C = (K − σ M)⁻¹ M` has eigenvalues `1/(λ − σ)`; for a stiff pencil
   (`E ≈ 2.1e5`, `ρ ≈ 7.8e-9`) the generalized eigenvalues are `λ ≈ 1e10 … 1e13`,
   so `C`'s eigenvalues — and the Lanczos `beta` — legitimately sit at `~1e-11`
   and trip the absolute threshold after ~6 steps. The subspace is spuriously
   capped: the two lowest eigenvalues come out right but residuals floor at
   `~1e-3` and higher modes are noise.
2. **No implicit restart** — the driver grows the Krylov dimension to `maxiter`
   then stops; clustered / near-degenerate low modes plateau above `tol` no
   matter how large the subspace.

## What Changes

Rewrite the `eigsh` Lanczos driver (no public API change) to:

- **Relative breakdown threshold** — declare an invariant subspace only when
  `beta` is negligible *relative to the operator scale* (a running max of the
  `alpha`/`beta` magnitudes), fixing the stiff-pencil spurious breakdown.
- **Internal operator rescaling** — scale the shift-invert operator by
  `s = trace(K)/trace(M)` so the Lanczos recurrence runs at O(1) magnitude
  (better conditioning for the projected dense `eigh`); shift back with
  `λ = σ + s/θ`. This removes the need for the caller-side pencil rescaling
  CalculiX++ uses today.
- **Thick-restart Lanczos** — when the wanted modes have not converged, restart
  by retaining the Ritz vectors nearest `σ` (their Ritz values become the
  diagonal, the last-`beta`×last-eigenvector-component become the arrowhead
  spike) plus the current residual, and continue. Converged/near-converged Ritz
  pairs are deflated and the rest keep refining across cycles, so clustered
  spectra converge. Full reorthogonalization keeps the basis M-orthonormal.

Convergence is still gated on the true generalized residual
`‖K x − λ M x‖ / ‖K x‖ < tol`.

## Capabilities

### Modified Capabilities
- `sparse`: the "Sparse generalized symmetric eigensolver" requirement gains a
  robustness guarantee — `eigsh` converges on stiff (large-magnitude-eigenvalue)
  and tightly-clustered SPD pencils, not only well-conditioned ones.

## Impact

- **Code**: rewrite `src/sparse/eigen.cpp` (relative breakdown, rescaling,
  thick-restart driver). No change to the public `eigsh` / `EigshResult` API;
  `maxiter` is reinterpreted as the maximum number of restart cycles (`0` picks a
  default) and `iterations` now reports the total Lanczos steps performed.
- **Tests**: add a **stiff** pencil (eigenvalues `~1e10`, exercising the relative
  threshold) and a **clustered** pencil (near-degenerate low modes, exercising
  thick restart) to `tests/oracle/generate.py`, validated against
  `scipy.sparse.linalg.eigsh`; update the `eigsh` cases in
  `tests/test_sparse.cpp`. The stiff case is a regression test for bug #1.
- **Dependencies**: none new.
</content>
