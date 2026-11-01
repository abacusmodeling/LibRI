# LibRI

Read this in other languages: [English](./README.md), [中文](./README.cn.md)


## RI technique

Conventional Density functional theory (DFT) methods are not sufficient to properly describe certain types of binding interactions.
It is often necessary to use higher-level methods like hybrid functionals, random-phase approximation (RPA), GW methods and beyond to accurately treat.
However, the computational cost of these methods are prohibitively expensive.

For local bases, invoking the resolution-of-the-identity (RI) significantly reduces the storage requirement and the computational cost.


## requirements for LibRI

LibRI is a C++ library using RI technique for methods beyond DFT. It is a head-only library.

These are necessary for LibRI:
- C++ compiler, supported C++11 and OpenMP.
- MPI library, used for inter-process communication.
- BLAS and LAPACK libraries, used as backends for Tensor operations.
  > If Math Kernel Library (MKL) is linked as the BLAS and LAPACK backend, it's recommended to define macro `__MKL_RI` before including any LibRI's header file. Some LibRI's functions will be silently substituted with calls to MKL routines.
- [cereal](https://uscilab.github.io/cereal/) library, a head-only library for serialization.
- [LibComm](https://github.com/abacusmodeling/LibComm.git) library, a head-only library for inter-process communication.

The optional DDLA GPU backend (`__GPU_RI` and `__DDLA_RI`) requires C++17 and
[LibDDLA 0.0.6 or newer](https://github.com/userchba/LibDDLA/releases/tag/v0.0.6)
built with CUDA or HIP. It uses the installed public headers; CMake consumers can
link `ddla::ddla` from `find_package(LibDDLA 0.0.6 CONFIG REQUIRED)`.
