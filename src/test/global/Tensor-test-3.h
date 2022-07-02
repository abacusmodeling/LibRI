#pragma once

#include "Tensor-test.h"
#include "test/print_stl.h"
#include <iostream>
#include <complex>

namespace Tensor_Test
{

	static Tensor<double> init_vector(const size_t n)
	{
		Tensor<double> t({n});
		for(size_t i=0; i<t.shape[0]; ++i)
			t(i) = i;
		return t;
	}

	static Tensor<double> init_matrix(const size_t nr, const size_t nc)
	{
		Tensor<double> t({nr,nc});
		for(size_t i0=0; i0<t.shape[0]; ++i0)
			for(size_t i1=0; i1<t.shape[1]; ++i1)
				t(i0,i1) = i0*10+i1;
		return t;
	}

	static void test_property()
	{
		const Tensor<double> t = init_matrix(3,2);

		std::cout<<t.shape<<std::endl;
		std::cout<<t.get_shape_all()<<std::endl;

		std::cout<<t<<std::endl;
		std::cout<<t.reshape({2,3})<<std::endl;
		std::cout<<t.transpose()<<std::endl;
		std::cout<<t.reshape({6})<<std::endl;
	}

	static void test_add()
	{
		std::cout<<init_matrix(2,3)+init_matrix(2,3)<<std::endl;
		// 0  2  4
		// 20 22 24
	}
	
	static void test_multiply()
	{
		const Tensor<double> v3 = init_vector(3);
		const Tensor<double> m23 = init_matrix(2,3);
		const Tensor<double> m34 = init_matrix(3,4);

		std::cout<<v3<<std::endl;
		std::cout<<v3*v3<<std::endl<<std::endl;
		// 5

		std::cout<<m23<<std::endl;
		std::cout<<v3<<std::endl;
		std::cout<<m23*v3<<std::endl<<std::endl;
		// 5 35

		std::cout<<v3<<std::endl;
		std::cout<<m34<<std::endl;
		std::cout<<v3*m34<<std::endl<<std::endl;
		// 50 53 56 59

		std::cout<<m23<<std::endl;
		std::cout<<m34<<std::endl;
		std::cout<<m23*m34<<std::endl<<std::endl;
		// 50  53  56  59
		// 350 383 416 449
	}

	static void test_norm()
	{
		Tensor<double> td({2});
		td(0)=3;	td(1)=-4;
		std::cout<<td.norm(1)<<std::endl;									// 7
		std::cout<<td.norm(2)<<std::endl;									// 5
		std::cout<<td.norm(3)<<std::endl;									// 4.497941445275415
		std::cout<<td.norm(std::numeric_limits<double>::max())<<std::endl;	// 4

		Tensor<std::complex<double>> tc({2});
		tc(0)={-3,4}; tc(1)={6,-8};
		std::cout<<tc.norm(1)<<std::endl;									// 15
		std::cout<<tc.norm(2)<<std::endl;									// 11.180339887498949
		std::cout<<tc.norm(3)<<std::endl;									// 10.400419115259519
		std::cout<<tc.norm(std::numeric_limits<double>::max())<<std::endl;	// 10
	}
}
