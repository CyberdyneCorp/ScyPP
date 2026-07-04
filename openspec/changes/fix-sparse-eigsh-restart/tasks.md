> GitHub issue: [#15](https://github.com/CyberdyneCorp/SciPP/issues/15)
>
> Single-PR change: make the `eigsh` Lanczos driver robust on stiff and
> clustered pencils. No public API change.

## 1. Relative breakdown + internal rescaling (bug #1)

- [x] 1.1 Rescale the shift-invert operator by `s = trace(K)/trace(M)` (guarded to a finite positive value, else `1`); apply `Ĉ v = s · (K − σ M)⁻¹ (M v)` and shift back with `λ = σ + s/θ`
- [x] 1.2 Replace the absolute `beta > 1e-12` invariant-subspace test with a relative one against a running operator scale (max `|alpha|` / `beta` seen)

## 2. Thick-restart Lanczos (bug #2)

- [x] 2.1 Extract an M-orthonormal Lanczos that extends a basis (with full reorthogonalization) and fills a dense projected matrix, supporting an arrowhead first step after restart
- [x] 2.2 On non-convergence, thick-restart: retain the `nkeep` Ritz vectors nearest `σ` (diagonal = Ritz values, arrowhead spike = `beta_last · s_i[last]`) plus the residual vector, then continue; loop up to `maxiter` cycles
- [x] 2.3 Keep convergence gated on the true generalized residual `‖K x − λ M x‖ / ‖K x‖`; reinterpret `maxiter` as max restart cycles and `iterations` as total Lanczos steps

## 3. Oracle + tests

- [x] 3.1 Regenerate/adjust golden data: a **stiff** pencil (λ ~ 1e10, from scaling the existing pencil) and a **clustered** pencil (near-degenerate low modes, block-diagonal bars) with `scipy.sparse.linalg.eigsh` references in `tests/oracle/generate.py`
- [x] 3.2 Tests in `tests/test_sparse.cpp`: stiff pencil converges (regression for #1), clustered pencil converges via restart, existing well-conditioned + nonzero-σ cases still pass; run clang + gcc + asan
- [x] 3.3 Update `README.md`, `CHANGELOG.md`, and the sparse spec (via archive)
</content>
