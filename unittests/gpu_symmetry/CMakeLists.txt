cmake_minimum_required(VERSION 3.18)
project(libri_gpu_symmetry_tests LANGUAGES CXX)

option(ENABLE_GPU "Run the DDLA/CUDA contraction checks" OFF)
set(LIBRI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "LibRI source tree")
set(LIBCOMM_ROOT "" CACHE PATH "LibComm installation")
set(CEREAL_ROOT "" CACHE PATH "cereal installation")
set(DDLA_ROOT "" CACHE PATH "DDLA installation")

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(test_lri_gpu_symmetry test_lri_gpu_symmetry.cpp)
target_compile_features(test_lri_gpu_symmetry PRIVATE cxx_std_14)
target_include_directories(test_lri_gpu_symmetry PRIVATE
  "${LIBRI_ROOT}/include" "${LIBCOMM_ROOT}/include" "${CEREAL_ROOT}/include")
target_link_libraries(test_lri_gpu_symmetry PRIVATE
  MPI::MPI_CXX OpenMP::OpenMP_CXX ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

if(ENABLE_GPU)
  find_package(CUDAToolkit REQUIRED)
  find_library(DDLA_LIBRARY NAMES ddla PATHS "${DDLA_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
  target_compile_features(test_lri_gpu_symmetry PRIVATE cxx_std_17)
  target_compile_definitions(test_lri_gpu_symmetry PRIVATE __GPU_RI __DDLA_RI DDLA_USE_CUDA)
  target_include_directories(test_lri_gpu_symmetry PRIVATE "${DDLA_ROOT}/include")
  target_link_libraries(test_lri_gpu_symmetry PRIVATE
    ${DDLA_LIBRARY} CUDA::cudart CUDA::cublas CUDA::curand CUDA::cusolver)
endif()

enable_testing()
add_test(NAME lri_symmetry COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
  ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_lri_gpu_symmetry> ${MPIEXEC_POSTFLAGS})
set_tests_properties(lri_symmetry PROPERTIES ENVIRONMENT
  "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
