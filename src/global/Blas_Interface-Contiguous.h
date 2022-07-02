// ===================
//  Author: Peize Lin
//  date: 2021.10.31
// ===================

#pragma once

#include "Blas_Interface.h"

namespace Blas_Interface
{
	// nrm2 = ||x||_2
	template<typename T>
	static inline T nrm2(const int n, const std::complex<T>*const X)
	{
		constexpr int incX = 1;
		return nrm2(n, X, incX);
	}
	template<typename T>
	static inline T nrm2(const int n, const T*const X)
	{
		constexpr int incX = 1;
		return nrm2(n, X, incX);
	}

	// d = Vx * Vy
	template<typename T>
	static inline double dot(const int n, const T*const X, const T*const Y)
	{
		constexpr int incX = 1;
		constexpr int incY = 1;
		return dot(n, X, incX, Y, incY);
	}

	// Vy = alpha * Ma.? * Vx + beta * Vy
	template<typename T>
	static inline void gemv(const char transA, const int m, const int n,
		const double alpha, const T*const A, const T*const X,
		const double beta, T*const Y)
	{
		const int ldA = n;
		constexpr int incX = 1;
		constexpr int incY = 1;
		gemv(transA, m, n,
			alpha, A, ldA, X, incX,
			beta, Y, incY);
	}

	// Mc = alpha * Ma.? * Mb.? + beta * Mc
	template<typename T>
	static inline void gemm(const char transA, const char transB, const int m, const int n, const int k,
		const double alpha, const T*const A, const T*const B,
		const double beta, T*const C)
	{
		const int ldA = (transA=='N') ? k : m;
		const int ldB = (transB=='N') ? n : k;
		const int ldC = n;
		gemm(transA, transB, m, n, k,
			alpha, A, ldA, B, ldB, 
			beta, C, ldC);
	}  

	// Mc = alpha * Ma   * Ma.T + beta * C
	// Mc = alpha * Ma.T * Ma   + beta * C
	template<typename T>
	static inline void syrk(const char uploC, const char transA, const int n, const int k,
		const double alpha, const T*const A,
		const double beta, T*const C)
	{
		const int ldA = (transA=='N') ? k : n;
		const int ldC = n;
		syrk(uploC, transA, n, k,
			alpha, A, ldA,
			beta, C, ldC);
	}
}