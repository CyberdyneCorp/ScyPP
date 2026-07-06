from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout


class SciPPConan(ConanFile):
    name = "scipp"
    version = "1.6.0"
    license = "MIT"
    description = "Modern C++20 port of SciPy, built on NumPP"
    settings = "os", "compiler", "build_type", "arch"

    options = {
        "with_blas": [True, False],
        "with_lapack": [True, False],
        "with_cuda": [True, False],
        "with_opencl": [True, False],
        "with_metal": [True, False],
    }
    default_options = {
        "with_blas": False,
        "with_lapack": False,
        "with_cuda": False,
        "with_opencl": False,
        "with_metal": False,
    }

    exports_sources = "CMakeLists.txt", "cmake/*", "include/*", "src/*", "tests/*"

    def requirements(self):
        # Pinned NumPP dependency (the array engine + device backends).
        self.requires("numpp/1.6.0")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        for opt in ("blas", "lapack", "cuda", "opencl", "metal"):
            tc.variables[f"SCIPP_WITH_{opt.upper()}"] = bool(getattr(self.options, f"with_{opt}"))
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["scipp"]
