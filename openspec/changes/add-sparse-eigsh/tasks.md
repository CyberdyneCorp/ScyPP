> GitHub issue: [#12](https://github.com/CyberdyneCorp/SciPP/issues/12)
>
> Single-PR change: a shift-invert Lanczos generalized symmetric eigensolver on
> top of the reusable sparse factorization, validated against SciPy's `eigsh`.

## 1. Reusable sparse factorization (factor once, solve many)

- [x] 1.1 Refactor `src/sparse/factor.cpp` into a `build_factorization` producing a `Factorization` (SPD → Cholesky, else LU) and an `apply_factorization`; rewrite `sparse_direct_solve`/`sparse_direct_factor_nnz` on top of it (no behavior change)
- [x] 1.2 Expose the opaque `Factorization` + `sparse_direct_factorize`/`factorization_ok`/`factorization_solve` in `include/scipp/sparse/detail.hpp`

## 2. Shift-invert Lanczos eigensolver

- [x] 2.1 Add `struct EigshResult { ndarray eigenvalues; ndarray eigenvectors; int iterations; bool converged; }` and `eigsh(K, M, k, sigma, tol, maxiter)` to `include/scipp/sparse/sparse.hpp`
- [x] 2.2 Implement `src/sparse/eigen.cpp`: form `A = K − σ M`, factor once, run M-orthonormal Lanczos on `C = (K − σ M)⁻¹ M` with full reorthogonalization; solve the projected tridiagonal with `numpp::linalg::eigh`; shift back `λ = σ + 1/θ`, select the `k` nearest `σ`, return ascending mass-normalized pairs
- [x] 2.3 Gate `converged` on the true generalized residual `‖K x − λ M x‖ / ‖K x‖ < tol`; grow the Krylov dimension up to the subspace cap until the `k` modes converge; report `iterations` (subspace dimension built)
- [x] 2.4 Register `sparse/eigen.cpp` in `src/CMakeLists.txt`

## 3. Dependency + oracle + tests

- [x] 3.1 Bump the NumPP pin to 1.6.0 (`find_package(NumPP 1.6.0)` and `conanfile.py`) for the O(n³) dense `eigh`
- [x] 3.2 Oracle: add a generalized SPD pencil (`K = tri(-1,2,-1)`, `M = tri(1,4,1)/6`, n=64) and `scipy.sparse.linalg.eigsh(K, M, sigma=0)` reference to `tests/oracle/generate.py`; freeze golden data
- [x] 3.3 Tests in `tests/test_sparse.cpp`: eigenvalues match SciPy, generalized residual small, eigenvectors mass-normalized, distinct modes M-orthogonal, `converged` true; run and pass (clang + ASan)
- [x] 3.4 Update `README.md`, `CHANGELOG.md`, and sparse docs for `eigsh`
</content>
