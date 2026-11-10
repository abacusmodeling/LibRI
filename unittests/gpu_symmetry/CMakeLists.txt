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
add_executable(test_gpu_resident test_gpu_resident.cpp)
foreach(test_target test_lri_gpu_symmetry test_gpu_resident)
target_compile_features(${test_target} PRIVATE cxx_std_14)
target_include_directories(${test_target} PRIVATE
  "${LIBRI_ROOT}/include" "${LIBCOMM_ROOT}/include" "${CEREAL_ROOT}/include")
target_link_libraries(${test_target} PRIVATE
  MPI::MPI_CXX OpenMP::OpenMP_CXX ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

if(ENABLE_GPU)
  find_package(LibDDLA 0.0.6 REQUIRED CONFIG PATHS
    "${DDLA_ROOT}/lib/cmake/LibDDLA" "${DDLA_ROOT}/lib64/cmake/LibDDLA"
    NO_DEFAULT_PATH)
  target_compile_features(${test_target} PRIVATE cxx_std_17)
  target_compile_definitions(${test_target} PRIVATE __GPU_RI __DDLA_RI DDLA_USE_CUDA)
  target_link_libraries(${test_target} PRIVATE ddla::ddla)
endif()

endforeach()

enable_testing()
add_test(NAME lri_symmetry COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
  ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_lri_gpu_symmetry> ${MPIEXEC_POSTFLAGS})
set_tests_properties(lri_symmetry PROPERTIES ENVIRONMENT
  "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")

# Two ranks exercise differing shared_ptr owner counts without introducing collectives.
foreach(backend cpu gpu)
  if(backend STREQUAL "gpu" AND NOT ENABLE_GPU)
    continue()
  endif()
  foreach(ranks 1 2)
    foreach(threads 1 4)
      set(test_name resident_${backend}_${ranks}mpi_${threads}omp)
      add_test(NAME ${test_name} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_gpu_resident> ${MPIEXEC_POSTFLAGS} ${backend})
      set_tests_properties(${test_name} PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77
        ENVIRONMENT "OMP_NUM_THREADS=${threads};OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    endforeach()
  endforeach()
endforeach()
