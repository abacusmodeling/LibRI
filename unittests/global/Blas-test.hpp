// ===================
//  Author: Peize Lin
//  date: 2021.10.31
// ===================

#pragma once

#include <vector>
#include <iostream>

#include "global/Blas_Interface.h"
#include "global/Blas_Interface-Contiguous.h"
#include "global/Blas_Interface-Tensor.h"
#include "global/Tensor.h"
#include "Tensor-test-3.hpp"

#include <mkl_blas.h>

namespace Blas_Test
{
	void nrm2()
	{
		const Tensor<double> v3 = Tensor_Test::init_vector(3);
		std::cout<<v3<<std::endl;
		std::cout<<Blas_Interface::nrm2(v3)<<std::endl;
		// 2.23607
	}

	void dot1()
	{
		std::vector<double> a = {1,2,3};
		std::vector<double> b = {4,5,6};
		std::cout<<Blas_Interface::dot(3, a.data(), 1, b.data(), 1)<<std::endl;
		// 32
	}

	void dot2()
	{
		const MKL_INT N=3, INC=1;
		std::vector<double> a = {1,2,3};
		std::vector<double> b = {4,5,6};
		std::cout<<DDOT(&N, a.data(), &INC, b.data(), &INC)<<std::endl;
		// 32
	}

	void gemv()
	{
		const Tensor<double> m23 = Tensor_Test::init_matrix(2,3);
		const Tensor<double> v2 = Tensor_Test::init_vector(2);
		const Tensor<double> v3 = Tensor_Test::init_vector(3);
		Tensor<double> vr2({2});
		Tensor<double> vr3({3});

		std::cout<<m23<<std::endl;
		std::cout<<v3<<std::endl;
		Blas_Interface::gemv('N', m23.shape[0], m23.shape[1], 1.0, m23.ptr(), v3.ptr(), 0.0, vr2.ptr());
		std::cout<<vr2<<std::endl;
		Blas_Interface::gemv('N', 1.0, m23, v3, 0.0, vr2);
		std::cout<<vr2<<std::endl;
		std::cout<<Blas_Interface::gemv('N', 1.0, m23, v3)<<std::endl;
		// 5 35

		std::cout<<m23.transpose()<<std::endl;
		std::cout<<v2<<std::endl;
		Blas_Interface::gemv('T', m23.shape[0], m23.shape[1], 1.0, m23.ptr(), v2.ptr(), 0.0, vr3.ptr());
		std::cout<<vr3<<std::endl;
		Blas_Interface::gemv('T', 1.0, m23, v2, 0.0, vr3);
		std::cout<<vr3<<std::endl;
		std::cout<<Blas_Interface::gemv('T', 1.0, m23, v2)<<std::endl;
		// 10 11 12
	}

	void syrk()
	{
		const Tensor<double> m = Tensor_Test::init_matrix(2,3);

		std::cout<<m<<std::endl;
		std::cout<<m.transpose()<<std::endl;
		Tensor<double> m_mT = Tensor<double>({2,2});
		Blas_Interface::syrk('U', 'N', m.shape[0], m.shape[1], 1.0, m.ptr(), m.shape[1], 0.0, m_mT.ptr(), m_mT.shape[0]);
		std::cout<<m_mT<<std::endl;
		Blas_Interface::syrk('U', 'N', m.shape[0], m.shape[1], 1.0, m.ptr(), 0.0, m_mT.ptr());
		std::cout<<m_mT<<std::endl;
		Blas_Interface::syrk('U', 'N', 1.0, m, 0.0, m_mT);
		std::cout<<m_mT<<std::endl;
		std::cout<<Blas_Interface::syrk('U', 'N', 1.0, m)<<std::endl;
		// 5 35
		// 0 365

		std::cout<<m.transpose()<<std::endl;
		std::cout<<m<<std::endl;
		Tensor<double> mT_m = Tensor<double>({3,3});
		Blas_Interface::syrk('U', 'T', m.shape[1], m.shape[0], 1.0, m.ptr(), m.shape[1], 0.0, m_mT.ptr(), mT_m.shape[0]);
		std::cout<<mT_m<<std::endl;
		Blas_Interface::syrk('U', 'T', m.shape[1], m.shape[0], 1.0, m.ptr(), 0.0, m_mT.ptr());
		std::cout<<mT_m<<std::endl;
		Blas_Interface::syrk('U', 'T', 1.0, m, 0.0, mT_m);
		std::cout<<mT_m<<std::endl;
		std::cout<<Blas_Interface::syrk('U', 'T', 1.0, m)<<std::endl;
		// 100 110 120
		// 0   122 134
		// 0   0   148

	}
}