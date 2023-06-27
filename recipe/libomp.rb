class Libomp < Formula
    desc "LLVM's OpenMP runtime library"
    homepage "https://openmp.llvm.org/"
    url "https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/openmp-16.0.6.src.tar.xz"
    sha256 "a2536f06373774f2829d892b72eea5f97bab20e69b26b03042d947f992eb8e38"
    license "MIT"

    livecheck do
      url :stable
      regex(/^llvmorg[._-]v?(\d+(?:\.\d+)+)$/i)
    end

    bottle do
      sha256 cellar: :any,                 arm64_ventura:  "5528ca5676f0cbb175e80a0d27dcf67d8001ed5025da743a464d8e22085a9b62"
      sha256 cellar: :any,                 arm64_monterey: "3dee22dd4f55d9bb85cc5b89ae5d99e6e2f52b151ca8768c9f71e41cf88f9986"
      sha256 cellar: :any,                 arm64_big_sur:  "c5c9a353ed022bd805478f108bdc62387f51ccbd4b9ca6bb2568c7e16f66bff3"
      sha256 cellar: :any,                 ventura:        "30cb6ac784eaa406e43e48102d4351cb2a2acbb10687ba2b8b5ab9327e860d81"
      sha256 cellar: :any,                 monterey:       "d7e7a3bc9015ad377aef9a642ca00c2e186aa740df58079cf68fafad884764e7"
      sha256 cellar: :any,                 big_sur:        "4ae172c013b17cde11b708443b02a88605c10789717d781de860f01737ec8e0b"
      sha256 cellar: :any_skip_relocation, x86_64_linux:   "c7c1b23acb19fe987543c327f246b4d26e1e917d9b3d2a8467e5c40d7e792117"
    end

    # # Ref: https://github.com/Homebrew/homebrew-core/issues/112107
    # keg_only "it can override GCC headers and result in broken builds"

    depends_on "cmake" => :build
    depends_on "lit" => :build
    uses_from_macos "llvm" => :build

    on_linux do
      depends_on "python@3.11"
    end

    resource "cmake" do
      url "https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/cmake-16.0.6.src.tar.xz"
      sha256 "39d342a4161095d2f28fb1253e4585978ac50521117da666e2b1f6f28b62f514"
    end

    def install
      (buildpath/"src").install buildpath.children
      (buildpath/"cmake").install resource("cmake")

      # Disable LIBOMP_INSTALL_ALIASES, otherwise the library is installed as
      # libgomp alias which can conflict with GCC's libgomp.
      args = ["-DLIBOMP_INSTALL_ALIASES=OFF"]
      args << "-DOPENMP_ENABLE_LIBOMPTARGET=OFF" if OS.linux?

      # Build universal binary
      ENV.permit_arch_flags
      ENV.runtime_cpu_detection
      args << "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64"

      system "cmake", "-S", "src", "-B", "build/shared", *std_cmake_args, *args
      system "cmake", "--build", "build/shared"
      system "cmake", "--install", "build/shared"

      system "cmake", "-S", "src", "-B", "build/static",
                      "-DLIBOMP_ENABLE_SHARED=OFF",
                      *std_cmake_args, *args
      system "cmake", "--build", "build/static"
      system "cmake", "--install", "build/static"
    end

    test do
      assert_equal version, resource("cmake").version, "`cmake` resource needs updating!"
      (testpath/"test.cpp").write <<~EOS
        #include <omp.h>
        #include <array>
        int main (int argc, char** argv) {
          std::array<size_t,2> arr = {0,0};
          #pragma omp parallel num_threads(2)
          {
              size_t tid = omp_get_thread_num();
              arr.at(tid) = tid + 1;
          }
          if(arr.at(0) == 1 && arr.at(1) == 2)
              return 0;
          else
              return 1;
        }
      EOS
      system ENV.cxx, "-Werror", "-Xpreprocessor", "-fopenmp", "test.cpp", "-std=c++11",
                      "-I#{include}", "-L#{lib}", "-lomp", "-o", "test"
      system "./test"
    end
  end

#   class Libomp < Formula
#   desc "LLVM's OpenMP runtime library"
#   homepage "https://openmp.llvm.org/"
#   url "https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/openmp-16.0.6.src.tar.xz"
#   sha256 "a2536f06373774f2829d892b72eea5f97bab20e69b26b03042d947f992eb8e38"
#   license "MIT"

#   livecheck do
#     url "https://llvm.org/"
#     regex(/LLVM (\d+\.\d+\.\d+)/i)
#   end

#   bottle do
#     sha256 cellar: :any,                 arm64_monterey: "cf1058b26e1a778e523d51562c99b4145aea1b1cb89f1c60b3315677a86c7a08"
#     sha256 cellar: :any,                 arm64_big_sur:  "bbf77a1a151f00a18e340ab1f655fb87fe787a85834518f1dc44bf0c52ae7d4c"
#     sha256 cellar: :any,                 monterey:       "e66d2009d6d205c19499dcb453dfac4376ab6bdba805987be00ddbbab65a0818"
#     sha256 cellar: :any,                 big_sur:        "ed9dc636a5fc8c2a0cfb1643f7932d742ae4805c3f193a9e56cab7d7cf7342e7"
#     sha256 cellar: :any,                 catalina:       "c72ce9beecde09052e7eac3550b0286ed9bfb2d14f1dd5954705ab5fb25f231b"
#     sha256 cellar: :any_skip_relocation, x86_64_linux:   "9fe14d5f4c8b472de1fad74278da6ba38da7322775b8a88ac61de0c373c4ad10"
#   end

#   depends_on "cmake" => :build
#   depends_on :xcode => :build # Sometimes CLT cannot build arm64
#   uses_from_macos "llvm" => :build

#   on_linux do
#     keg_only "provided by LLVM, which is not keg-only on Linux"
#   end

#   def install
#     # Disable LIBOMP_INSTALL_ALIASES, otherwise the library is installed as
#     # libgomp alias which can conflict with GCC's libgomp.

#     args = ["-DLIBOMP_INSTALL_ALIASES=OFF"]
#     args << "-DOPENMP_ENABLE_LIBOMPTARGET=OFF" if OS.linux?

#     # Build universal binary
#     ENV.permit_arch_flags
#     ENV.runtime_cpu_detection
#     args << "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64"

#     system "cmake", "-S", "openmp-#{version}.src", "-B", "build/shared", *std_cmake_args, *args
#     system "cmake", "--build", "build/shared"
#     system "cmake", "--install", "build/shared"

#     system "cmake", "-S", "openmp-#{version}.src", "-B", "build/static",
#                     "-DLIBOMP_ENABLE_SHARED=OFF",
#                     *std_cmake_args, *args
#     system "cmake", "--build", "build/static"
#     system "cmake", "--install", "build/static"
#   end

#   test do
#     (testpath/"test.cpp").write <<~EOS
#       #include <omp.h>
#       #include <array>
#       int main (int argc, char** argv) {
#         std::array<size_t,2> arr = {0,0};
#         #pragma omp parallel num_threads(2)
#         {
#             size_t tid = omp_get_thread_num();
#             arr.at(tid) = tid + 1;
#         }
#         if(arr.at(0) == 1 && arr.at(1) == 2)
#             return 0;
#         else
#             return 1;
#       }
#     EOS
#     system ENV.cxx, "-Werror", "-Xpreprocessor", "-fopenmp", "test.cpp", "-std=c++11",
#                     "-L#{lib}", "-lomp", "-o", "test"
#     system "./test"
#   end
# end
