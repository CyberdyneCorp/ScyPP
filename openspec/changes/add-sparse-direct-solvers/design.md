## Context

`scipp::sparse` (Phase 9) delivered COO/CSR/CSC, SpMV/SpMM with backend dispatch,
matrix-free `cg`/`gmres`, and csgraph. The linear-solver path, however, is a
placeholder: `spsolve` densifies (`numpp::linalg::solve(A.toarray(), b)`) and
`cg`/`gmres` are unpreconditioned with no convergence reporting. Issue #10
documents the consequence on FE stiffness systems: 95 s / 0.5 GB for a 8268-DOF
direct solve, and silent ~16 % error from non-converged CG.

Constraints:
- Header-light, C++20, no new external dependencies. NumPP dense `solve` is
  available and used only as the small-N fallback.
- Tests are frozen SciPy golden data (`tests/oracle/generate.py` →
  `tests/golden/golden.hpp`); every new numeric path needs an oracle.
- Existing `sp_A` (4×4) golden values must not change; existing public
  signatures (`spsolve`, `cg`, `gmres`, `norm`) stay source-compatible.
- Data model: matrices live as CSR; factorizations are most naturally expressed
  in CSC. Conversion helpers are needed in `detail.hpp`.

## Goals / Non-Goals

**Goals:**
- Genuinely-sparse direct solve: O(nnz(L)) memory, not O(N²).
- SPD Cholesky/LDLᵀ (the FE case) + general Gilbert–Peierls LU.
- RCM fill-reducing ordering by default; ordering selection enum.
- Preconditioned CG/GMRES (Jacobi, IC(0), ILU(0)) with an `IterationResult` and
  explicit non-convergence signaling.
- Reusable factorization objects (`splu`/`factorized`/`spilu`).
- Matrix diagnostics (nnz, bandwidth, symmetry).
- SciPy oracle parity for every new path.

**Non-Goals:**
- AMD/METIS ordering in the first PRs (RCM is the default; AMD is a later,
  in-tree addition — no external METIS).
- Supernodal/multifrontal performance engineering; correctness and asymptotic
  sparsity come first.
- Eigensolvers (`eigsh`/`eigs`/`svds`) — they consume the factorization objects
  built here but remain tracked in `add-sparse-extras`.
- Parallel/GPU factorization.

## Decisions

**D1 — CSparse-style algorithms, implemented from the published algorithm, not
copied.** Use Tim Davis' CSparse formulations (Gilbert–Peierls left-looking LU
with depth-first reachability triangular solve; up-looking Cholesky with
elimination tree + column counts) as the algorithmic reference. They are compact
(~a few hundred lines total), well-documented, and asymptotically sparse.
Alternative considered: right-looking multifrontal — rejected as far more code
for no correctness benefit at this stage.

**D2 — Factor in CSC.** Triangular solves and both factorizations are column
algorithms. Convert the incoming CSR to CSC once (transpose of a symmetric
pattern is free; general transpose is O(nnz)). Add `detail` helpers to move
between CSR/CSC and raw `int64/double` arrays.

**D3 — RCM ordering by default, behind an `OrderingMethod` enum
(`Natural`, `Rcm`, `Amd`).** RCM is ~50 lines (BFS from a pseudo-peripheral node
via degree, reverse the level order), bandwidth-reducing, and highly effective on
banded FE stiffness. Apply the permutation symmetrically (`P A Pᵀ`) for Cholesky,
and as a column/row order for LU. `Amd` is declared now but may return
`Natural`/`Rcm` until its PR lands (documented). Alternative: AMD-first — rejected
because AMD is much larger to implement and verify; RCM unblocks FE immediately.

**D4 — SPD detection = try Cholesky, fall back to LU.** `spsolve` attempts
Cholesky when the pattern is structurally symmetric; a non-positive-definite
pivot aborts Cholesky (reported, not silently wrong) and the caller retries with
LU. This avoids a separate, unreliable SPD test.

**D5 — Small-N dense fallback.** Below a threshold (default `N ≤ 64`, tunable),
`spsolve` keeps the dense `numpp::linalg::solve` path — the 4×4 `sp_A` oracle
stays on this path so its golden values are unchanged, and tiny systems avoid
factorization overhead.

**D6 — Additive API; reporting via a new struct, not a signature break.** Keep
`ndarray cg(A,b,tol,maxiter)` and `gmres(...)`. Add:
- `enum class Preconditioner { None, Jacobi, IC0, ILU0 };`
- `enum class OrderingMethod { Natural, Rcm, Amd };`
- `struct IterationResult { ndarray x; int iterations; double final_residual; bool converged; };`
- reporting overloads `cg(A,b, Preconditioner, tol, maxiter) -> IterationResult`
  and the GMRES analogue (name chosen to avoid overload ambiguity — likely
  `cg_report`/`gmres_report`, finalized in Phase 1).
- factorization classes `SparseLU` / `SparseCholesky` with `.solve(b)`, plus
  free functions `splu(A, ordering)`, `factorized(A)`, `spilu(A)`.

**D7 — Preconditioners.** Jacobi = inverse diagonal (trivial). IC(0)/ILU(0) =
incomplete factorization restricted to the sparsity pattern of `A` (no fill),
reusing the numeric-factorization inner loop with a "drop everything outside the
pattern" rule. Applied as `M⁻¹ r` via the existing triangular solves.

**D8 — Phasing.** Each phase is a shippable PR with its own oracle tests:
Phase 1 (preconditioners + reporting), Phase 2 (ordering + direct factorization +
`spsolve` routing), Phase 3 (reusable objects + `spilu`), Phase 4 (diagnostics).
Phase 1 lands first because silent non-convergence is a correctness hazard.

## Risks / Trade-offs

- **Numeric parity vs SciPy (SuperLU/UMFPACK internals differ)** → Compare
  `A @ x ≈ b` and `x ≈ spsolve` at a documented tolerance (e.g. rtol 1e-8) rather
  than bit-exact factors; pivoting/ordering differences are absorbed by solving
  the same system.
- **RCM under-reduces fill vs AMD on unstructured meshes** → Acceptable for
  banded FE now; `OrderingMethod::Amd` reserved for the follow-up. Report factor
  nnz in diagnostics so users can see the cost.
- **Cholesky on a matrix that is symmetric in pattern but indefinite** → Detect
  non-positive pivot, abort with a clear failure, fall back to LU (D4).
- **Ill-conditioned FE systems still hard for iterative solvers even
  preconditioned** → IC(0) is a mild preconditioner; document that the direct
  path is the robust default and CG+IC(0) is the memory-lean alternative. Report
  non-convergence honestly (the core fix).
- **Integer index width** (`int64` throughout) → keep `int64_t` to match existing
  CSR arrays; watch for `int` truncation in inner loops.
- **New golden data** must be regenerated with SciPy available → gate on the
  existing `generate.py` workflow; commit the frozen header so CI needs no Python.

## Migration Plan

Additive and phased — no rollback of existing behavior. After each phase:
regenerate golden data (`python3 tests/oracle/generate.py`), review the diff,
build and run `tests/test_sparse.cpp`. `spsolve` routing change (Phase 2) is the
only behavioral change to an existing entry point; the 4×4 oracle stays on the
dense fallback path so its result is identical, and larger systems get a new
oracle proving `A @ x ≈ b`. If a phase regresses, revert that phase's PR
independently.

## Open Questions

- Final names for the reporting entry points (`cg_report` vs an options struct) —
  decided in Phase 1 implementation to keep overload resolution unambiguous.
- Whether `factorized` returns a `std::function`-style functor or the concrete
  `SparseLU` — leaning concrete class with an `operator()`/`.solve` for zero-cost
  reuse.
