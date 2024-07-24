########## CMake Configuration #########
# Below are the suggested configurations to CMake
set(BUILD_SHARED_LIBS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_COMPILER_LAUNCHER ccache)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

########## RAF Configuration ##########
# Convention: RAF_USE_LIB could be ON/OFF or a string indicating path to LIB

# RAF_USE_LLVM. Option: [ON/OFF/Path-to-llvm-config-executable]"
set(RAF_USE_LLVM llvm-config-8)
set(HIDE_PRIVATE_SYMBOLS ON)

# RAF_USE_GTEST. Option: [ON/OFF]
set(RAF_USE_GTEST ON)

# RAF_USE_CUDA. Option: [ON/OFF]
set(RAF_USE_CUDA ON)

# RAF_USE_CUBLAS. Option: [ON/OFF]
set(RAF_USE_CUBLAS ON)

# RAF_USE_CUDNN. Option: [ON/OFF/Path-To-CUDNN]. You may use environment variables, like $ENV{CUDNN_HOME}
set(RAF_USE_CUDNN ON)

# RAF_USE_SANITIZER. Option: [OFF/ASAN/MSAN/TSAN/UBSAN]"
set(RAF_USE_SANITIZER OFF)

# RAF_USE_MPI. Option: [ON/OFF]
set(RAF_USE_MPI ON)

# RAF_USE_NCCL. Option: [ON/OFF]
set(RAF_USE_NCCL ON)
set(NCCL_ROOT_DIR /opt/nccl/build)

# RAF_USE_ORTOOLS. Option: [ON/OFF/Path-To-ORTOOLS]
set(RAF_USE_ORTOOLS /opt/or-tools_Ubuntu-18.04-64bit_v9.3.10497)

set(RAF_CUDA_ARCH 70)
set(RAF_USE_CUTLASS OFF)
set(CMAKE_BUILD_TYPE Debug)
