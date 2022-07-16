// ===================
//  Author: Peize Lin
//  date: 2021.10.31
// ===================

#pragma once

#include "Blas-Fortran.h"

#include <string>
#include <stdexcept>

namespace Blas_Interface
{	
	static inline char change_uplo(const char &uplo)
	{
		switch(uplo)
		{
			case 'U': return 'L';
			case 'L': return 'U';
			default: throw std::invalid_argument("trans must be 'U' or 'L'.\nIn "+std::string(__FILE__)+" line "+std::to_string(__LINE__)+". ");
		}
	}

	static inline char change_trans_NT(const char &trans)
	{
		switch(trans)
		{
			case 'N': return 'T';
			case 'T': return 'N';
			default: throw std::invalid_argument("trans must be 'N' or 'T'.\nIn "+std::string(__FILE__)+" line "+std::to_string(__LINE__)+". ");
		}
	}

	// nrm2 = ||x||_2
	static inline double nrm2(const int n, const double*const X, const int incX)
	{
		return dnrm2_(&n, X, &incX);
	}
	static inline double nrm2(const int n, const std::complex<double>*const X, const int incX)
	{
		return dznrm2_(&n, X, &incX);
	}

	// d = Vx * Vy
	static inline double dot(const int n, const double*const X, const int incX, const double*const Y, const int incY)
	{
		return ddot_(&n, X, &incX, Y, &incY);
	}

	// Vy = alpha * Ma.? * Vx + beta * Vy
	static inline void gemv(const char transA, const int m, const int n,
		const double alpha, const double*const A, const int ldA, const double*const X, const int incX,
		const double beta, double*const Y, const int incY)
	{
		const char transA_changed = change_trans_NT(transA);
		dgemv_(&transA_changed, &n, &m,
			&alpha, A, &ldA, X, &incX,
			&beta, Y, &incY);
	}

	// Mc = alpha * Ma.? * Mb.? + beta * Mc
	static inline void gemm(const char transA, const char transB, const int m, const int n, const int k,
		const float alpha, const float*const A, const int ldA, const float*const B, const int ldB, 
		const float beta, float*const C, const int ldC)
	{
		sgemm_(&transB, &transA, &n, &m, &k,
			&alpha, B, &ldB, A, &ldA, 
			&beta, C, &ldC);
	}
	static inline void gemm(const char transA, const char transB, const int m, const int n, const int k,
		const double alpha, const double*const A, const int ldA, const double*const B, const int ldB, 
		const double beta, double*const C, const int ldC)
	{
		dgemm_(&transB, &transA, &n, &m, &k,
			&alpha, B, &ldB, A, &ldA, 
			&beta, C, &ldC);
	}
	static inline void gemm(const char transA, const char transB, const int m, const int n, const int k,
		const std::complex<float> alpha, const std::complex<float>*const A, const int ldA, const std::complex<float>*const B, const int ldB, 
		const std::complex<float> beta, std::complex<float>*const C, const int ldC)
	{
		cgemm_(&transB, &transA, &n, &m, &k,
			&alpha, B, &ldB, A, &ldA, 
			&beta, C, &ldC);
	}
	static inline void gemm(const char transA, const char transB, const int m, const int n, const int k,
		const std::complex<double> alpha, const std::complex<double>*const A, const int ldA, const std::complex<double>*const B, const int ldB, 
		const std::complex<double> beta, std::complex<double>*const C, const int ldC)
	{
		zgemm_(&transB, &transA, &n, &m, &k,
			&alpha, B, &ldB, A, &ldA, 
			&beta, C, &ldC);
	}	

	// Mc = alpha * Ma   * Ma.T + beta * C
	// Mc = alpha * Ma.T * Ma   + beta * C
	static inline void syrk(const char uploC, const char transA, const int n, const int k,
		const double alpha, const double*const A, const int ldA,
		const double beta, double*const C, const int ldC)
	{
		const char uploC_changed = change_uplo(uploC);
		const char transA_changed = change_trans_NT(transA);
		dsyrk_(&uploC_changed, &transA_changed, &n, &k,
			&alpha, A, &ldA,
			&beta, C, &ldC);
	}
}