# SciPP

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.25%2B-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20OpenCL%20%7C%20Metal-76B900?logo=nvidia&logoColor=white)](#tiered-acceleration)
[![Spec](https://img.shields.io/badge/spec-OpenSpec-3B5526)](openspec/)

Modern **C++20 port of [SciPy](https://github.com/scipy/scipy)** — the scientific
computing layer that sits on top of NumPy. SciPP is a clean-room implementation
validated against real SciPy as a numerical oracle, and is the SciPy sibling of:

- **[NumPP](https://github.com/CyberdyneCorp/NumPP)** — the NumPy port (N-D
  `ndarray`, dtypes, ufuncs, linalg, fft, random) with a tiered CPU / BLAS / GPU
  acceleration layer. **SciPP is built on NumPP** — it is the array engine.
- **[SymPP](https://github.com/leonardoaraujosantos/SymPP)** — the SymPy port
  (symbolic math).

```cpp
#include "scipp/scipp.hpp"
#include "numpp/core/ndarray.hpp"
using numpp::ndarray;
namespace sp = scipp::special;
namespace cst = scipp::constants;

// Special functions — scalar or element-wise over a numpp::ndarray.
double g   = sp::gamma(0.5);                  // 1.7724538509...
double b   = sp::jv(0.0, 2.0);                // Bessel J0(2)
ndarray ls = sp::logsumexp(x, /*axis=*/1);    // numerically stable

// Physical constants & conversions (CODATA, matches scipy.constants).
double c   = cst::c;                          // speed of light
double m_e = cst::value("electron mass");     // CODATA table lookup
double f   = cst::convert_temperature(100.0, "Celsius", "Fahrenheit");  // 212

// Linear algebra (scipy.linalg conventions, GEMM via NumPP).
auto [P, L, U] = scipp::linalg::lu(Asq);      // explicit P·L·U
ndarray E      = scipp::linalg::expm(Asq);    // matrix exponential
```

> **All 12 subpackages are implemented** (`special`, `constants`, `linalg`, `fft`,
> `optimize`, `integrate`, `differentiate`, `interpolate`, `stats`, `signal`, `odr`,
> `sparse`, `spatial`, `ndimage`, `cluster`, `io`). Deferred long-tail features are
> tracked as open OpenSpec backlog changes under `openspec/changes/`.

## Architecture

SciPP is the science layer: each subpackage takes `numpp::ndarray` in and out, and
delegates all array math, BLAS/LAPACK, and GPU work to the **NumPP** engine, which
in turn routes every operation through a single runtime dispatch that selects the
best available backend and always falls back to the portable CPU kernel. Real SciPy
is the numerical oracle every result is validated against.

```mermaid
flowchart TB
    subgraph SCIPP["SciPP - scipp:: (C++20 port of SciPy)"]
        MATH["special - constants - linalg - fft/fftpack"]
        NUM["optimize - integrate - differentiate - interpolate"]
        DATA["stats - signal - odr - cluster - io"]
        STRUCT["sparse - spatial - ndimage"]
    end

    subgraph NUMPP["NumPP engine - numpp::"]
        ARR["ndarray - dtypes - ufuncs"]
        NLA["linalg - fft - random"]
    end

    REG["Runtime dispatch: CapabilityRegistry - weak GpuVTable - last_backend()"]

    subgraph BACK["Backends - selected at runtime"]
        CPU["Portable CPU kernel (always present)"]
        BLAS["BLAS / LAPACK"]
        GPU["CUDA - OpenCL - Metal"]
    end

    MATH --> ARR
    NUM --> ARR
    DATA --> ARR
    STRUCT --> ARR
    ARR --> NLA
    NLA --> REG
    REG --> CPU
    REG --> BLAS
    REG --> GPU

    SCIPY["SciPy 1.15 - numerical oracle"] -. validates .-> SCIPP

    style SCIPP fill:#0B5394,color:#fff
    style NUMPP fill:#1155CC,color:#fff
    style REG fill:#B45309,color:#fff
    style SCIPY fill:#38761D,color:#fff
```

## Why SciPP

SciPy is the de-facto scientific toolbox, but it is Python + Cython + Fortran and
assumes a desktop/server with BLAS/LAPACK. SciPP brings the same algorithms to a
single portable C++20 source tree that:

- **Builds and runs everywhere, including iOS and Android** — a portable,
  dependency-free C++ CPU kernel is always present and is the only thing required
  to build (only NumPP + the standard library).
- **Accelerates SciPy's CPU-bound hot kernels on the GPU** — FFT, dense linalg,
  sparse SpMV, `ndimage` separable convolution and pairwise distances gain
  optional **CUDA / OpenCL / Metal** backends, routed through NumPP's weak-linked
  device dispatch and always falling back to the CPU.
- **Is validated against real SciPy** — every numeric result is asserted
  `allclose` to the reference implementation.

## Tiered acceleration

SciPP does not build a second device stack. It reuses NumPP's runtime dispatch —
`CapabilityRegistry`, `last_backend()`, the `NUMPP_GPU_TARGET` override
(`cpu|cuda|opencl|metal|auto`) and the device buffer pool — and registers
SciPy-specific kernels into the same weak-linked vtable.

```
scipp::linalg / fft / signal / optimize / stats / sparse / spatial / ndimage / ...
                         │  (numpp::ndarray in, numpp::ndarray out)
                         ▼
   numpp::ndarray · dtypes · ufuncs · linalg · fft · random          ← NumPP engine
                         │
         CapabilityRegistry · weak GpuVTable · last_backend()
                         │
   portable CPU kernel ─► BLAS/LAPACK ─► CUDA / OpenCL / Metal  (runtime select)
```

Below a per-operation size threshold, or for unsupported dtypes, or when no device
is present, the portable CPU kernel is used. An accelerated result always equals
the CPU result within documented floating-point tolerance.

## Target capability map

Each capability ports a public SciPy subpackage into the `scipp::<name>`
namespace. See [`openspec/project.md`](openspec/project.md) for the full map.

| Subpackage | Scope |
|------------|-------|
| `special` · `constants` | Special functions (gamma/erf/Bessel/orthogonal polys), CODATA constants |
| `linalg` | Decompositions, matrix functions, solvers (BLAS/LAPACK + GPU GEMM) |
| `fft` · `fftpack` | FFT, real/N-D transforms, DCT/DST (GPU-accelerated) |
| `optimize` · `odr` | `minimize`, `root`, `least_squares`/`curve_fit`, `linprog`/`milp`; orthogonal-distance (total-least-squares) regression |
| `integrate` · `differentiate` | `quad`, `solve_ivp`, `solve_bvp`, finite differences |
| `interpolate` | `interp1d`, splines, `griddata`/`RBFInterpolator` |
| `stats` | Distributions, hypothesis tests, QMC, `gaussian_kde` |
| `signal` | Filtering, filter design, spectral analysis, LTI systems |
| `sparse` | CSR/CSC/COO formats, `sparse.linalg` (sparse-direct Cholesky/LU + preconditioned CG/GMRES + shift-invert `eigsh`), `csgraph` (GPU SpMV) |
| `spatial` | KD-trees, distances, ConvexHull/Delaunay/Voronoi, rotations |
| `ndimage` | Filters, morphology, measurements (GPU separable convolution) |
| `cluster` · `io` · `datasets` | k-means/hierarchical, Matrix Market/WAV/ARFF I/O, sample datasets |

## Project status

**v1.6 — all 12 phases shipped.** Every public SciPy subpackage's commonly-used
surface is ported, built on NumPP and validated against SciPy 1.15 — **146 cases /
7710 oracle checks, 0 divergences**:

- **Phase 1** — `scipp::special` (gamma/erf/Bessel/exponential integrals/
  orthogonal evaluators/combinatorics/`logsumexp`/`softmax`) and `scipp::constants`
  (CODATA table + scale constants + unit conversions). This phase also stood up the
  foundation: CMake/C++20 skeleton, pinned NumPP `find_package` dependency, the
  `scipp::error` model, and the frozen-golden SciPy oracle harness.
- **Phase 2** — `scipp::linalg`: `inv`/`det`/`solve`/`lstsq`/`pinv`/`pinvh`/`norm`,
  `lu`/`qr`/`svd`/`cholesky` (+ factor/solve helpers), `eig`/`eigh`, `expm` (Padé)
  and `polar`, and special-matrix constructors. Standard decompositions delegate to
  NumPP; GEMM-heavy paths route through NumPP's BLAS/GPU `matmul`.
- **Phase 3** — `scipp::fft` (+ legacy `scipp::fftpack`): the FFT family
  (`fft`/`rfft`/`hfft`, N-D variants, `fftfreq`/`fftshift`) delegated to NumPP, plus
  SciPy's **DCT/DST** types I–IV, `next_fast_len`, and axis-wise transforms.
- **Phase 4** — `scipp::optimize`: scalar root finders (`brentq`/`bisect`/`newton`),
  `minimize_scalar` (brent/bounded), `minimize` (Nelder–Mead/BFGS), and nonlinear
  least squares (`least_squares`/`curve_fit`/`fsolve`).
- **Phase 5** — `scipp::integrate` + `scipp::differentiate`: fixed-sample and
  adaptive quadrature (`trapezoid`/`simpson`/`quad`/`fixed_quad`), explicit ODE
  solvers (`solve_ivp` RK45/RK23), and `derivative`/`jacobian`/`hessian`.
- **Phase 6** — `scipp::interpolate`: `Interp1d`, `CubicSpline`/`PchipInterpolator`/
  `Akima1DInterpolator` (with derivatives), `RegularGridInterpolator`/`interpn`, and
  `RBFInterpolator` (radial basis functions).
- **Phase 7** — `scipp::stats`: continuous distributions (`norm`/`gamma`/`beta`/`t`/
  `f`/`chi2`/…) with `pdf`/`cdf`/`ppf`, summary statistics, correlation/regression
  (`pearsonr`/`spearmanr`/`linregress`), hypothesis tests (`ttest_*`/`f_oneway`/
  `ks_2samp`/`chi2_contingency`/`normaltest`), and `gaussian_kde`.
- **Phase 8** — `scipp::signal`: convolution, windows, waveforms, time-domain
  filtering (`lfilter`/`filtfilt`/`sosfilt`/`hilbert`/`freqz`), filter design
  (`butter`/`cheby1`/`cheby2`/`ellip`/`bessel`/`firwin` + zpk conversions), spectral
  estimation (`welch`/`periodogram`/`csd`/`coherence`/`spectrogram`/`stft`), peak
  analysis (`find_peaks`/`peak_prominences`/`peak_widths`), LTI systems
  (`bode`/`step`/`impulse`/`lsim`), resampling (`resample`/`resample_poly`/
  `decimate`/`upfirdn`), and `savgol_filter`/`medfilt`/`wiener`/`convolve2d`.
- **Phase 9** — `scipp::sparse`: COO/CSR/CSC formats + conversions, `eye`/`diags`,
  `spmv`/`spmm` with NumPP capability-registry **backend dispatch** (CPU kernel +
  GPU-ready architecture), `sparse.linalg` (`spsolve`/`cg`/`gmres`/`norm`), and
  `sparse.csgraph` (`dijkstra`/`bellman_ford`/`floyd_warshall`/
  `connected_components`/`minimum_spanning_tree`).
  - **Sparse direct solvers** — `spsolve` factors genuinely sparsely rather than
    densifying: an up-looking **Cholesky** for SPD systems (e.g. FE stiffness) and
    a Gilbert–Peierls **LU** with partial pivoting for general systems, both with
    a fill-reducing **RCM** ordering (`OrderingMethod::{Natural,Rcm,Amd}`; AMD
    reserved). A dense fallback is kept for very small `N`. `factor_nnz(A,
    ordering)` reports the factor fill for tuning.
  - **Preconditioned iterative solvers** — `cg`/`gmres` accept a
    `Preconditioner` (`None`/`Jacobi`/`IC0`/`ILU0`); the `cg_report`/`gmres_report`
    variants return an `IterationResult { x, iterations, final_residual,
    converged }` and **signal non-convergence** instead of returning a
    silently-wrong vector at `maxiter`.
  - **Generalized symmetric eigensolver** — `eigsh(K, M, k, sigma)` solves the
    sparse pencil `K x = λ M x` (SPD `M`) for the lowest-`k` (or nearest-`sigma`)
    modes via **thick-restart shift-invert Lanczos**: `(K − σ M)` is factored
    **once** (reusing the sparse direct factorization) and applied across an
    `M`-orthonormal Lanczos, with the small projected problem solved by NumPP's
    dense `eigh`. The operator is internally rescaled and the invariant-subspace
    test is relative to its scale, so **stiff** pencils (`λ ~ 1e10`+) don't break
    down spuriously; **thick restart** deflates converged Ritz pairs so
    **clustered / near-degenerate** spectra converge. Returns an
    `EigshResult { eigenvalues, eigenvectors, iterations, converged }` with
    ascending eigenvalues and mass-normalized (`xᵀ M x = 1`) eigenvectors — the
    scalable path for FE modal / frequency analysis. The same engine is exposed as
    `eigsh_gen(A, B, k, sigma)` for a general symmetric pencil `A x = θ B x` with
    **indefinite `A`** and **SPD `B`** (the nearest-`sigma` primitive).
  - **Linear-buckling eigensolver** — `eigsh_buckling(K, K_geo, k)` returns the `k`
    **smallest positive** load factors `λ` of `(K + λ K_geo) φ = 0` (elastic `K`
    SPD, geometric `K_geo` symmetric-indefinite) — the `*BUCKLE` eigenproblem. It
    reduces to `K_geo φ = μ (K φ)`, `μ = −1/λ` (so the critical factor is the
    *most-negative* `μ`, not the one nearest `σ = 0`) and runs an **adaptive-σ
    walk** driven by cheap factor-only definiteness probes (a free Sturm bit:
    `(K_geo − σ K)` is SPD ⇔ `σ` is below the whole spectrum) to place the shift
    below all modes, then a single `eigsh_gen` solve. Returns a `BucklingResult
    { load_factors, modes, iterations, shifts, converged }` with ascending positive
    load factors and `K`-normalized modes — the scalable path for FE buckling analysis.
- **Phase 10** — `scipp::spatial`: distances (`pdist`/`cdist`/`squareform` + metrics
  with backend dispatch), `KDTree`, 2-D `ConvexHull`/`Delaunay`, and 3-D rotations
  (`transform::Rotation` quat/matrix/euler/rotvec + `apply`/`inv`/compose/`Slerp`).
- **Phase 11** — `scipp::ndimage`: filters (`gaussian`/`uniform`/`median`/`sobel`/…
  with boundary modes + dispatch), morphology (binary/grey + `distance_transform_edt`),
  measurements (`label`/`center_of_mass`), and order-0/1 geometric transforms
  (`shift`/`zoom`/`rotate`/`affine_transform`/`map_coordinates`).
- **Phase 12** — `scipp::cluster` (`whiten`/`vq`/`kmeans2`, `linkage`/`fcluster`/
  `cophenet`) and `scipp::io` (Matrix Market `mmread`/`mmwrite`, WAV `wavread`/
  `wavwrite`, ARFF `loadarff`).

The architecture, CUDA/OpenCL/Metal backend strategy, and the full parity roadmap
are specified with **OpenSpec** under [`openspec/`](openspec/). Each remaining
subpackage graduates into its own OpenSpec change, ported against the SciPy oracle.

```bash
openspec list                              # active changes
openspec show bootstrap-scipp-foundation   # the foundation + parity roadmap
openspec validate --all --strict           # validate the specs
```

## Building

SciPP depends on NumPP via `find_package(NumPP)`. Developer tasks are driven by a
[`justfile`](justfile) (install [`just`](https://github.com/casey/just)), matching
the NumPP/SymPP workflow — run `just` to list every recipe:

```bash
# 1. Build + install the pinned NumPP dependency into .deps/ (expects ../NumPP)
just bootstrap                 # or: just bootstrap /path/to/NumPP

# 2. Build + run the SciPy-oracle test suite (CPU-only, mobile-friendly)
just test                      # configure + build + run scipp_tests
just ctest                     # the same suite through CTest

# Other common recipes
just gpu-detect                # probe this host for CUDA/OpenCL/Metal, recommend a flag
just build                     # configure + compile library and tests
just debug                     # Debug build with assertions
just gcc                       # build + test with GCC
just asan                      # build + test under ASan/UBSan
just oracle                    # regenerate frozen golden data (needs python3 + scipy)
just spec                      # openspec validate --all --strict
just ci                        # local CI: clang tests + gcc + spec
just clean                     # remove all build dirs
```

Plain CMake still works if you prefer it (`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`).

For **GPU acceleration**, probe the host first and then pass the recommended backend
flag through `just configure` (requires a NumPP package built with the matching
backend):

```bash
just gpu-detect                        # Linux/macOS/Windows: reports CUDA/OpenCL/Metal + a flag
just configure -DSCIPP_WITH_CUDA=ON    # or -DSCIPP_WITH_OPENCL=ON / -DSCIPP_WITH_METAL=ON
just build
```

Backend feature flags (`SCIPP_WITH_{BLAS,LAPACK,CUDA,OPENCL,METAL}`) all default
**OFF**; enabling one enables the matching NumPP backend so both layers share a
single device runtime. The frozen oracle data lets CI run the suite without Python;
refresh it with `just oracle` after changing the test set.

## Relationship to the reference SciPy

The upstream SciPy checkout is the **oracle and reference implementation**; SciPP
re-implements its algorithms in portable C++ (no CPython C-API, no `PyObject`) and
adds GPU acceleration where SciPy has none. Spec requirements cite the SciPy source
they port as breadcrumbs, e.g. `(oracle: scipy/linalg/_decomp_lu.py)`.

## Consuming an installed SciPP

SciPP installs a `find_package(SciPP)` CONFIG package that exports the namespaced
`scipp::scipp` target and re-resolves its NumPP dependency transitively — no vendoring:

```bash
cmake --install build --prefix /your/prefix     # after a build
```

```cmake
# downstream CMakeLists.txt
find_package(SciPP CONFIG REQUIRED)
target_link_libraries(your_app PRIVATE scipp::scipp)
```

Point `CMAKE_PREFIX_PATH` at the SciPP prefix (and the NumPP prefix it depends on).
The library is also packaged for Conan (`conanfile.py`) and vcpkg (`vcpkg.json`).

## Versioning & API stability

SciPP follows **[semantic versioning](https://semver.org/)**. The API is still settling:
minor releases may carry breaking changes until the project commits to long-term API
stability. The shared library carries a semver `SOVERSION` (`libscipp.so.<major>`) so
consumers can pin ABI compatibility, and the version is declared once in `CMakeLists.txt`
and kept in sync across `conanfile.py` and `vcpkg.json`. Notable changes are tracked in
[CHANGELOG.md](CHANGELOG.md).

## Contributing

Work is spec-driven. Before writing code, read the relevant spec in `openspec/specs/` and
open changes through the OpenSpec workflow. Every bug fix ships with a regression test. See
**[CONTRIBUTING.md](CONTRIBUTING.md)** for the full workflow, the
[Code of Conduct](CODE_OF_CONDUCT.md), and [security reporting](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE).
