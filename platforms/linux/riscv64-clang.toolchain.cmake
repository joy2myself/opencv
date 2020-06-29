set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(triple riscv64-unknown-linux-gnu)
set(clang_build_dir /home/git/rvv-llvm/build)
set(riscv_toolchain_dir /home/RISCV)

set(CMAKE_C_COMPILER   ${clang_build_dir}/bin/clang)
set(CMAKE_C_COMPILER_TARGET   ${triple})
set(CMAKE_CXX_COMPILER ${clang_build_dir}/bin/clang++)
set(CMAKE_CXX_COMPILER_TARGET ${triple})

set(CMAKE_SYSROOT ${riscv_toolchain_dir}/sysroot/)

set(C_FLAGS "-march=rv64gcv --gcc-toolchain=${riscv_toolchain_dir} -w")
set(CXX_FLAGS "-march=rv64gcv --gcc-toolchain=${riscv_toolchain_dir} -w")

string(REPLACE ";" " " CMAKE_C_FLAGS "${C_FLAGS}")
string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CXX_FLAGS}")