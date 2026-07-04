# SciPP developer tasks — run `just` (or `just --list`) to see everything.
# Requires: cmake >= 3.25, clang++ or g++, ninja (for the gcc/asan recipes),
# a pinned NumPP install under .deps/numpp (see `just bootstrap`), and
# python3 + scipy (only for `just oracle`, to regenerate the frozen golden data).

build_dir := "build"
cxx       := "clang++"
numpp_src := "../NumPP"
numpp_lib := ".deps/numpp/lib"

# Show the available recipes (default).
default:
    @just --list

# Build + install the pinned NumPP dependency into .deps/numpp so that
# find_package(NumPP) resolves it. Pass a source path to override ../NumPP.
bootstrap src=numpp_src:
    ./scripts/bootstrap_numpp.sh {{src}}

# Configure the CPU-only Release build. Pass extra cmake flags, e.g.
#   just configure -DSCIPP_WITH_CUDA=ON
configure *FLAGS:
    cmake -S . -B {{build_dir}} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER={{cxx}} {{FLAGS}}

# Best-effort probe (no build; Linux, macOS, Windows) that recommends the matching
# configure flag, falling back to the always-available CPU backend. SciPP's GPU
# backends come from NumPP, so a recommended backend also needs a NumPP built with
# the matching NUMPP_WITH_* flag (the default `just bootstrap` builds a CPU-only
# NumPP). On Windows run this from Git Bash / MSYS2 / WSL (bash must be on PATH).
# Detect usable GPU backends on this host (CUDA / OpenCL / Metal), or CPU-only.
gpu-detect:
    #!/usr/bin/env bash
    set -uo pipefail
    os=$(uname -s); arch=$(uname -m)
    case "$os" in
      MINGW*|MSYS*|CYGWIN*) plat=Windows ;;
      Darwin)               plat=macOS   ;;
      Linux)                plat=Linux   ;;
      *)                    plat="$os"   ;;
    esac
    echo "Host: $plat ($os $arch)"
    echo

    cuda=no; opencl=no; metal=no

    # --- CUDA (NVIDIA) — needs the driver at runtime; nvcc to build ------------
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
      drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
      nvcc=$(command -v nvcc >/dev/null 2>&1 && nvcc --version | grep -oE 'release [0-9.]+' | head -1 || echo "toolkit not found")
      echo "✅ CUDA   : ${gpu:-NVIDIA GPU} (driver ${drv:-?}, ${nvcc})"
      cuda=yes
    else
      echo "❌ CUDA   : no NVIDIA driver (nvidia-smi) detected"
    fi

    # --- OpenCL — clinfo if present, else probe the ICD loader / framework -----
    if command -v clinfo >/dev/null 2>&1; then
      n=$(clinfo -l 2>/dev/null | grep -ciE 'device|platform' || true)
      if [ "${n:-0}" -gt 0 ]; then
        dev=$(clinfo -l 2>/dev/null | grep -iE 'device' | head -1 | sed 's/^[[:space:]]*//')
        echo "✅ OpenCL : ${dev:-device present} (clinfo)"; opencl=yes
      else
        echo "❌ OpenCL : clinfo found no devices"
      fi
    elif [ "$plat" = macOS ] && [ -d /System/Library/Frameworks/OpenCL.framework ]; then
      echo "✅ OpenCL : Apple OpenCL.framework present (install clinfo for details)"; opencl=yes
    elif [ "$plat" = Windows ] && { [ -f /c/Windows/System32/OpenCL.dll ] || reg query "HKLM\\SOFTWARE\\Khronos\\OpenCL\\Vendors" >/dev/null 2>&1; }; then
      echo "✅ OpenCL : Windows OpenCL ICD present (install clinfo to enumerate devices)"; opencl=yes
    elif ls /etc/OpenCL/vendors/*.icd >/dev/null 2>&1 || ldconfig -p 2>/dev/null | grep -q libOpenCL; then
      echo "✅ OpenCL : ICD loader present (install clinfo to enumerate devices)"; opencl=yes
    else
      echo "❌ OpenCL : no ICD loader / OpenCL runtime detected"
    fi

    # --- Metal (Apple GPU) — Apple platforms only ------------------------------
    if [ "$plat" = macOS ]; then
      dev=$(system_profiler SPDisplaysDataType 2>/dev/null | grep -iE 'Chipset Model|Metal' | head -1 | sed 's/^[[:space:]]*//')
      if [ -d /System/Library/Frameworks/Metal.framework ]; then
        echo "✅ Metal  : ${dev:-Metal framework present}"; metal=yes
      else
        echo "❌ Metal  : Metal.framework not found"
      fi
    else
      echo "❌ Metal  : Apple platforms only"
    fi

    echo
    if [ "$cuda" = yes ]; then rec="CUDA    →  just configure -DSCIPP_WITH_CUDA=ON   (needs a CUDA-enabled NumPP)"
    elif [ "$metal" = yes ]; then rec="Metal   →  just configure -DSCIPP_WITH_METAL=ON  (needs a Metal-enabled NumPP)"
    elif [ "$opencl" = yes ]; then rec="OpenCL  →  just configure -DSCIPP_WITH_OPENCL=ON (needs an OpenCL-enabled NumPP)"
    else rec="CPU only →  just build   (portable CPU backend; always available)"; fi
    echo "Recommended: $rec"

# Compile the library and the test suite.
build: configure
    cmake --build {{build_dir}} -j

# Compile only the library.
lib: configure
    cmake --build {{build_dir}} --target scipp -j

# Run the SciPy-oracle test suite (against the frozen golden data).
test: build
    LD_LIBRARY_PATH={{numpp_lib}} ./{{build_dir}}/tests/scipp_tests
alias unit := test

# Run the test suite through CTest.
ctest: build
    ctest --test-dir {{build_dir}} --output-on-failure

# Configure + build + test with debug symbols and assertions (separate dir).
debug:
    cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER={{cxx}}
    cmake --build build-debug -j
    LD_LIBRARY_PATH={{numpp_lib}} ./build-debug/tests/scipp_tests

# Build + test with GCC (separate build dir).
gcc:
    cmake -S . -B build-gcc -G Ninja -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release
    cmake --build build-gcc -j
    LD_LIBRARY_PATH={{numpp_lib}} ./build-gcc/tests/scipp_tests

# Build + test under AddressSanitizer / UBSan (separate build dir).
asan:
    cmake -S . -B build-asan -G Ninja -DCMAKE_CXX_COMPILER={{cxx}} -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
    cmake --build build-asan -j
    LD_LIBRARY_PATH={{numpp_lib}} ./build-asan/tests/scipp_tests

# Regenerate the frozen oracle golden data (requires python3 + scipy).
oracle:
    python3 tests/oracle/generate.py

# Validate all OpenSpec specs and changes.
spec:
    openspec validate --all --strict

# Full local CI: clang tests + gcc + spec.
ci: test gcc spec

# Remove all build directories.
clean:
    rm -rf {{build_dir}} build-debug build-gcc build-asan
