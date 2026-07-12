> GitHub issue: [#18](https://github.com/CyberdyneCorp/SciPP/issues/18)
>
> Single-PR change: a sparse smallest-positive-load-factor buckling eigensolver
> `eigsh_buckling` plus the reusable indefinite-`A`/SPD-`B` generalized primitive
> `eigsh_gen`, both layered on the existing thick-restart Lanczos engine and
> validated against a dense `scipy.linalg.eigh(K_geo, K)` oracle.

## 1. Free Sturm bit from the existing factorization

- [x] 1.1 Add `bool factorization_definite(const std::shared_ptr<Factorization>& F)` (returns `F && F->ok && F->used_cholesky`) to `src/sparse/factor.cpp` and declare it in `include/scipp/sparse/detail.hpp`

## 2. Generalized primitive (option b) via a shared core

- [x] 2.1 Refactor the `eigsh` body in `src/sparse/eigen.cpp` into an anonymous-namespace `eigsh_core(const CsrMatrix& A, const CsrMatrix& B, int k, double sigma, double tol, int maxiter)` (mechanical `K→A`, `M→B` rename; `TRLanczos`/`Operator`/`select_nearest`/`gen_residual` unchanged), and make public `eigsh` a one-line `return eigsh_core(K, M, …);`
- [x] 2.2 Robustify the operator scale in `eigsh_core` to `s = (finite(trA,trB) && trB > 0) ? std::max(std::fabs(trA), trB) / trB : 1.0` (identical to the old value for every SPD pencil where `trA ≥ trB`; keeps `s` well-scaled when `A = K_geo` has non-positive trace)
- [x] 2.3 Add public `EigshResult eigsh_gen(const CsrMatrix& A, const CsrMatrix& B, int k, double sigma = 0.0, double tol = 1e-8, int maxiter = 0)` in `include/scipp/sparse/sparse.hpp`, implemented as `return eigsh_core(A, B, …);`

## 3. Buckling driver (option a) — adaptive-σ walk

- [x] 3.1 Add `struct BucklingResult { ndarray load_factors; ndarray modes; int iterations = 0; int shifts = 0; bool converged = false; }` and `BucklingResult eigsh_buckling(const CsrMatrix& K, const CsrMatrix& K_geo, int k, double sigma0 = 0.0, double tol = 1e-8, int maxiter = 0)` to `include/scipp/sparse/sparse.hpp`
- [x] 3.2 Implement file-local helpers in `src/sparse/eigen.cpp` (each ≤ 15 cognitive complexity): `frob_scale(K, K_geo)`; `probe_definite(K, K_geo, sigma, shifts&)` (build `S = K_geo.add(K.scaled(-sigma))`, factor, return `factorization_definite`); `bracket_below_spectrum(...)` → `{sigma_lo (definite), sigma_hi (indefinite)}`; `refine_shift(...)` (few bisections, loose stop at magnitude-ratio `< 2`) → `sigma_star`; `collect_positive(mu, vecs, k)` → `{lambda ascending, modes}`
- [x] 3.3 Implement `eigsh_buckling`: guards (`k ≤ 0 || n ≤ 0` → empty; clamp `k ≤ n`); seed shift from `sigma0` or `frob_scale`; `bracket_below_spectrum` + `refine_shift`; single `eigsh_gen(K_geo, K, kp = min(n, k + max(k, 4)), sigma_star, tol, maxiter)`; on a singular final factor nudge `sigma_star *= 1.1` once and retry; `collect_positive`; set `converged = R.converged && (#positive ≥ k)`, `iterations = R.iterations`, `shifts` = probe count
- [x] 3.4 Confirm `src/sparse/eigen.cpp` stays registered in `src/CMakeLists.txt` (no new TU)

## 4. Oracle + tests

- [x] 4.1 Oracle: add to `tests/oracle/generate.py` two buckling pencils via `mu = sla.eigh(Kgeo, K, eigvals_only=True); lam = np.sort([-1.0/t for t in mu if t < 0])` — closed-form Euler beam `K = [[4,2],[2,4]]`, `K_geo = -(1/30)[[4,-1],[-1,4]]` (λ = {12, 60}) and discriminating indefinite `K = [[2,.5,0],[.5,2,.5],[0,.5,2]]`, `K_geo = diag(-0.5, 1/3, -0.2)` (λ = {3.837, 9.766}); emit `sp_buckle_K/_Kgeo/_lambda` and `sp_buckle2_K/_Kgeo/_lambda`; regenerate `tests/golden/golden.hpp`
- [x] 4.2 Tests in `tests/test_sparse.cpp`: for each pencil `CHECK_CLOSE(load_factors[i], golden::sp_buckle*_lambda[i], 1e-7, 1e-9)`, per-mode `‖K φ + λ K_geo φ‖ / ‖K φ‖ < 1e-7`, `φᵀ K φ ≈ 1`, assert **every** returned load factor `> 0`, and `converged = true`; add an `eigsh_gen` case that reproduces the discriminating pencil's most-negative μ
- [x] 4.3 Build and run green on clang + gcc + ASan (`just test`, `just gcc`, `just asan`)

## 5. Version + docs

- [x] 5.1 Bump `CMakeLists.txt` project `VERSION 1.4.0 → 1.5.0`
- [x] 5.2 Update `README.md` (sparse eigensolver bullet + `BucklingResult`), `CHANGELOG.md` (`## 1.5.0` entry), and any sparse doc block for `eigsh_buckling`/`eigsh_gen`