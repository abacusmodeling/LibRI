// ===================
//  Author: Peize Lin
//  date: 2021.10.31
// ===================

#pragma once

#include <complex>

extern "C"
{
	// nrm2 = ||x||_2
	double dnrm2_(const int*const n, const double*const X, const int*const incX);
	double dznrm2_(const int*const n, const std::complex<double>*const X, const int*const incX);

	// d = Vx * Vy
	double ddot_(const int*const n, const double*const X, const int*const incX, const double*const Y, const int*const incY);

	// Vy = alpha * Ma.? * Vx + beta * Vy
	double dgemv_(const char*const transA, const int*const m, const int*const n,
		const double*const alpha, const double*const A, const int*const ldA, const double*const X, const int*const incX,
		const double*const beta, double*const Y, const int*const incY);

	// Mc = alpha * Ma.? * Mb.? + beta * Mc
	void sgemm_(const char*const transA, const char*const transB, const int*const m, const int*const n, const int*const k,
		const float*const alpha, const float*const A, const int*const ldA, const float*const B, const int*const ldB, 
		const float*const beta, float*const C, const int*const ldC);
	void dgemm_(const char*const transA, const char*const transB, const int*const m, const int*const n, const int*const k,
		const double*const alpha, const double*const A, const int*const ldA, const double*const B, const int*const ldB, 
		const double*const beta, double*const C, const int*const ldC);
	void cgemm_(const char*const transA, const char*const transB, const int*const m, const int*const n, const int*const k,
		const std::complex<float>*const alpha, const std::complex<float>*const A, const int*const ldA, const std::complex<float>*const B, const int*const ldB, 
		const std::complex<float>*const beta, std::complex<float>*const C, const int*const ldC);
	void zgemm_(const char*const transA, const char*const transB, const int*const m, const int*const n, const int*const k,
		const std::complex<double>*const alpha, const std::complex<double>*const A, const int*const ldA, const std::complex<double>*const B, const int*const ldB, 
		const std::complex<double>*const beta, std::complex<double>*const C, const int*const ldC);

	// Mc = alpha * Ma   * Ma.T + beta * C
	// Mc = alpha * Ma.T * Ma   + beta * C
	void dsyrk_(const char*const uploC, const char*const transA, const int*const n, const int*const k,
		const double*const alpha, const double*const A, const int*const ldA,
		const double*const beta, double*const C, const int*const ldC);
}