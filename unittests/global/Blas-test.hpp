// ===================
//  Author: Peize Lin
//  date: 2021.10.31
// ===================

#pragma once

#include <vector>
#include <iostream>

#include "RI/global/Blas_Interface.h"
#include "RI/global/Blas_Interface-Contiguous.h"
#include "RI/global/Blas_Interface-Tensor.h"
#include "RI/global/Tensor.h"
#include "../global/Tensor-test-3.hpp"

#include <mkl_blas.h>

namespace Blas_Test
{
	template<typename Tdata>
	void nrm2()
	{
		const Tensor<Tdata> v3 = Tensor_Test::init_vector<Tdata>(3);
		std::cout<<v3<<std::endl;
		std::cout<<Blas_Interface::nrm2(v3)<<std::endl;
		// 2.23607
	}

	template<typename Tdata>
	void axpy()
	{
		const Tensor<Tdata> vx = Tensor_Test::init_vector<Tdata>(3);
		Tensor<Tdata> vy = Tensor_Test::init_vector<Tdata>(3);
		Blas_Interface::axpy(vx.shape[0], 2, vx.ptr(), 1, vy.ptr(), 1);
		std::cout<<vy<<std::endl;
		// 0, 3, 6
		Tensor<Tdata> vz = Blas_Interface::axpy(Tdata(2), vx);
		std::cout<<vz<<std::endl;
		// 0, 2, 4
	}

	template<typename Tdata>
	void dot_1()
	{
		std::vector<Tdata> a = {1,2,3};
		std::vector<Tdata> b = {4,5,6};
		std::cout<<Blas_Interface::dot(3, a.data(), 1, b.data(), 1)<<std::endl;
		// 32
	}

	void dot_2()
	{
		const MKL_INT N=3, INC=1;
		std::vector<double> a = {1,2,3};
		std::vector<double> b = {4,5,6};
		std::cout<<DDOT(&N, a.data(), &INC, b.data(), &INC)<<std::endl;
		// 32
	}

	template<typename Tdata>
	void dot_complex()
	{
		const std::vector<Tdata> a = {{1,2},{3,4}};
		const std::vector<Tdata> b = {{5,6},{7,8}};
//		std::cout<<a<<std::endl<<b<<std::endl;
		std::cout<<Blas_Interface::dotu(a.size(), a.data(), 1, b.data(), 1)<<std::endl;
		/* -18+68i */
		std::cout<<Blas_Interface::dotc(a.size(), a.data(), 1, b.data(), 1)<<std::endl;
		/* 70-8i */
	}

	template<typename Tdata>
	void gemv()
	{
		const Tensor<Tdata> m23 = Tensor_Test::init_matrix<Tdata>(2,3);
		const Tensor<Tdata> v2 = Tensor_Test::init_vector<Tdata>(2);
		const Tensor<Tdata> v3 = Tensor_Test::init_vector<Tdata>(3);
		Tensor<Tdata> vr2({2});
		Tensor<Tdata> vr3({3});

		std::cout<<m23<<std::endl;
		std::cout<<v3<<std::endl;
		Blas_Interface::gemv('N', m23.shape[0], m23.shape[1], Tdata{1.0}, m23.ptr(), v3.ptr(), Tdata{0.0}, vr2.ptr());
		std::cout<<vr2<<std::endl;
		Blas_Interface::gemv('N', Tdata{1.0}, m23, v3, Tdata{0.0}, vr2);
		std::cout<<vr2<<std::endl;
		std::cout<<Blas_Interface::gemv('N', Tdata{1.0}, m23, v3)<<std::endl;
		// 5 35

		std::cout<<m23.transpose()<<std::endl;
		std::cout<<v2<<std::endl;
		Blas_Interface::gemv('T', m23.shape[0], m23.shape[1], Tdata{1.0}, m23.ptr(), v2.ptr(), Tdata{0.0}, vr3.ptr());
		std::cout<<vr3<<std::endl;
		Blas_Interface::gemv('T', Tdata{1.0}, m23, v2, Tdata{0.0}, vr3);
		std::cout<<vr3<<std::endl;
		std::cout<<Blas_Interface::gemv('T', Tdata{1.0}, m23, v2)<<std::endl;
		// 10 11 12
	}

	template<typename Tdata>
	void gemv_complex()
	{
		const Tensor<Tdata> m23 = Tensor_Test::init_matrix<Tdata>(2,3);
		const Tensor<Tdata> v3 = Tensor_Test::init_vector<Tdata>(3);
		Tensor<Tdata> vr2 = Tensor_Test::init_vector<Tdata>(2);
		std::cout<<m23<<std::endl;
		std::cout<<v3<<std::endl;
		std::cout<<vr2<<std::endl;
		Blas_Interface::gemv('N', Tdata{1.0,2.0}, m23, v3, Tdata{3,-4}, vr2);
		std::cout<<vr2<<std::endl;
		// (5,10)	(38,66)
	}

	void syrk()
	{
		const Tensor<double> m = Tensor_Test::init_matrix<double>(2,3);

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

	void test_all()
	{
		nrm2<float>();
		nrm2<double>();
		nrm2<std::complex<float>>();
		nrm2<std::complex<double>>();
		axpy<float>();
		axpy<double>();
		axpy<std::complex<float>>();
		axpy<std::complex<double>>();
		dot_1<float>();
		dot_1<double>();
		dot_2();
		dot_complex<std::complex<float>>();
		dot_complex<std::complex<double>>();
		gemv<float>();
		gemv<double>();
		gemv<std::complex<float>>();
		gemv<std::complex<double>>();
		gemv_complex<std::complex<float>>();
		gemv_complex<std::complex<double>>();
		syrk();
	}	
}