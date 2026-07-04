## Context

Linear buckling analysis needs the smallest positive load factors λ of
`(K + λ K_geo) φ = 0` — `K` elastic stiffness (SPD), `K_geo` geometric stiffness
(symmetric, typically indefinite). SciPP already has a thick-restart shift-invert
Lanczos generalized eigensolver (`eigsh`, #12/#15) whose recurrence runs in the
`B`-inner product and only requires **`B` SPD** — the first operand may be
indefinite. The buckling pencil supplies an SPD inner-product matrix (`B = K`), so
the engine applies with no numerical rewrite. The problem is one of **target
selection**, not a new kernel: "smallest positive λ" is not "nearest σ", and
naive `sigma = 0` returns the wrong (largest) load factor plus spurious negatives.

## Goals / Non-Goals

- **Goals**: the k smallest **positive** load factors ascending with K-normalized
  modes; a reusable indefinite-`A`/SPD-`B` nearest-σ primitive (`eigsh_gen`);
  reuse `TRLanczos` verbatim; one Lanczos solve after cheap factor-only shift
  location; zero change to `eigsh`/`EigshResult`; match a dense
  `eigh(K_geo, K)` oracle.
- **Non-Goals**: certified global inertia counts (LDLᵀ) for interior buckling
  bands; a multi-shift sweep; non-symmetric `K_geo`; the ARPACK mode-4
  load-factor-space transform (rejected — it would require a new operator, a new
  back-transform `λ = σθ/(θ−s)`, and a duplicated cycle loop; the `μ = −1/λ`
  pencil reuses the engine's native `λ = σ + s/θ` back-transform with only a
  post-map).

## Decisions

### Layering: shared core → generalized primitive → buckling driver
Today's `eigsh` body is refactored into `eigsh_core(A, B, k, σ, tol, maxiter)`
(mechanical `K/M → A/B` rename). `eigsh(K, M, …)` and the new public
`eigsh_gen(A, B, …)` both delegate; existing `sp_eig_*`/`sp_clus_*` goldens cover
the refactor. `eigsh_buckling` is a thin driver that owns all buckling policy and
calls `eigsh_gen` on the pencil `(K_geo, K)`.

### The transform and the sign (highest-consequence correctness point)
`(K + λ K_geo) φ = 0 ⟺ K_geo φ = μ (K φ)` with `μ = −1/λ`. Thus `λ > 0 ⟺ μ < 0`,
and over `μ < 0` the map `λ = −1/μ` is **monotone increasing** in μ. So:
- the **smallest positive** λ = the **algebraically most-negative** μ;
- `μ` ascending (all negative) ⇒ `λ = −1/μ` ascending (all positive) — the filter
  preserves order for free.

Verified: discriminating pencil `μ = {−0.2606, −0.1024, +0.1785}` →
`λ = {3.837, 9.766}`; most-negative μ = −0.2606 → λ = 3.837 (correct), whereas
μ nearest 0 = −0.1024 → λ = 9.766 (the wrong answer a `sigma = 0` target gives).

### Free Sturm bit via the Cholesky boundary
Because `B = K` is SPD, generalized Sylvester inertia gives
`(K_geo − σK) ≻ 0 ⇔ every μ > σ ⇔ σ` lies **below the whole spectrum**. The sparse
factorizer already attempts Cholesky only for SPD input and records
`used_cholesky`, so `factorization_definite(F) := F->ok && F->used_cholesky` is
exactly `#{μ < σ} == 0` — no LDLᵀ / inertia machinery needed. Verified on the
discriminating pencil: SPD at σ = −1.0 / −0.30 / −0.2606 (below μ_min = −0.2606),
NOT SPD at σ = −0.25 / −0.10 / 0.0.

### Adaptive-σ walk (cheap probes, one solve)
1. Seed σ (negative) from `sigma0` (`σ = −1/sigma0`) or `−‖K_geo‖_F/‖K‖_F`.
2. **Bracket** with factor-only `probe_definite`: obtain `σ_lo` (definite,
   `< μ_min`) and `σ_hi` (indefinite, `> μ_min`) by geometric expansion
   (double toward −∞ if the seed is indefinite; halve toward 0 if the seed is
   already definite). Terminates because `K_geo + |σ|K → SPD` as σ → −∞.
3. **Refine** by a few bisections, stopping loosely at magnitude-ratio
   `|σ_lo|/|σ_hi| < 2`, so `σ* = σ_lo` sits ~20–50 % below μ_min — close enough
   for strong shift-invert amplification, far enough for a well-conditioned SPD
   Cholesky factor in the production solve.
4. **One** `eigsh_gen(K_geo, K, kp, σ*)` with `kp = min(n, k + max(k, 4))`. Since
   `σ* < μ_min`, the k nearest-σ* eigenvalues are the k most-negative μ, ascending.
5. `collect_positive`: keep `μ_i < 0`, map `λ_i = −1/μ_i` (already ascending),
   carry the `K`-normalized modes; take the first `k`.

The expensive Lanczos runs exactly once; bracketing spends only ~3–6 sparse
factorizations (reported as `shifts`).

### Robustness grafts
- **Scale**: `eigsh_core` uses `s = max(|trace(A)|, trace(B))/trace(B)` (guarded
  `trace(B) > 0`) instead of the old `trace(A)/trace(B)`-with-`≤0→1` fallback. For
  every shipped SPD pencil `trace(K) ≥ trace(M)` so this equals the old value
  (no behavior change; the operator scale cancels in `λ = σ + s/θ`), while keeping
  `s` well-scaled when `A = K_geo` has non-positive trace.
- **Singular ridge**: if the production factor at `σ*` is singular (σ* grazed a
  pencil eigenvalue), nudge `σ* *= 1.1` (further below the spectrum) once and
  refactor; else `converged = false`.
- **Belt-and-suspenders**: the `kp` buffer guarantees ≥ k positive survivors after
  filtering even when a few near-boundary μ come back non-negative.
- **Convergence semantics**: `converged = eigsh_gen.converged && (#positive ≥ k)`,
  so a run that captured only k−1 compressive modes reports `false`.

### Cognitive complexity
`eigsh_buckling` is a short orchestration; the walk is split into `frob_scale`,
`probe_definite`, `bracket_below_spectrum`, `refine_shift`, `collect_positive`
(each ≤ 15, most ≤ 10), within the backend target.

## Risks / Trade-offs

- **No global inertia guarantee**: the definiteness bit only answers "is σ below
  the whole spectrum", which is all the smallest-positive walk needs; it cannot
  count interior modes. A future interior-band buckling search would want a real
  LDLᵀ inertia count. Mitigated here by placing σ* below all positive loads, where
  "k nearest" provably equals "k smallest positive".
- **Sign inversion is the highest-consequence bug**; the two committed unit
  pencils are chosen to fail loudly (returning 9.766 or a negative −5.603 instead
  of 3.837) if the map/filter is inverted, plus a hard "no returned λ ≤ 0" assert.
- **Symmetry**: `factorization_definite` relies on the Cholesky path, gated by the
  factorizer's `is_symmetric` (1e-12 relative) test; `(K_geo − σK)` is assembled
  as a single `.add()` to stay numerically symmetric. Callers must pass
  numerically symmetric `K`, `K_geo` (as `eigsh` already requires).
- **Clustered μ near μ_min** can slow the single solve; thick-restart already
  handles clustering (`sp_clus_*` precedent). A nudge-closer re-solve is a
  documented follow-up, not v1.

## Migration

Additive. `eigsh`/`EigshResult` unchanged. No dependency change (NumPP 1.6.0
already pinned). Version 1.4.0 → 1.5.0.