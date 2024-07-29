// ===================
//  Author: Peize Lin
//  date: 2023.08.03
// ===================

#pragma once

#include "Tensor_Multiply-test.hpp"

#define FOR_0(i,N)					for(std::size_t i=0; i<N; ++i)

namespace Tensor_Multiply_Test
{
	// Txy(x0,y2) = Tx(x0,a,b) * Ty(a,b,y2)
	template<typename Tdata>
	void x0y2_x0ab_aby2_test()
	{
		const std::size_t X0=2, Y2=3, A=4, B=5;
		const RI::Tensor<Tdata> Tx = init_tensor<Tdata>({X0,A,B});
		const RI::Tensor<Tdata> Ty = init_tensor<Tdata>({A,B,Y2});
		const RI::Tensor<Tdata> Txy = RI::Tensor_Multiply::x0y2_x0ab_aby2(Tx,Ty);
		FOR_0(x0,X0)
			FOR_0(y2,Y2)
			{
				Tdata result = 0;
				FOR_0(a,A)
					FOR_0(b,B)
						result += Tx(x0,a,b) * Ty(a,b,y2);
				assert(Txy(x0,y2) == result);
			}
	}

	// Txy(x2,y0) = Tx(a,b,x2) * Ty(y0,a,b)
	template<typename Tdata>
	void x2y0_abx2_y0ab_test()
	{
		const std::size_t X2=2, Y0=3, A=4, B=5;
		const RI::Tensor<Tdata> Tx = init_tensor<Tdata>({A,B,X2});
		const RI::Tensor<Tdata> Ty = init_tensor<Tdata>({Y0,A,B});
		const RI::Tensor<Tdata> Txy = RI::Tensor_Multiply::x2y0_abx2_y0ab(Tx,Ty);
		FOR_0(x2,X2)
			FOR_0(y0,Y0)
			{
				Tdata result = 0;
				FOR_0(a,A)
					FOR_0(b,B)
						result += Tx(a,b,x2) * Ty(y0,a,b);
				assert(Txy(x2,y0) == result);
			}
	}

	// Txy(x2,y2) = Tx(a,b,x2) * Ty(a,b,y2)
	template<typename Tdata>
	void x2y2_abx2_aby2_test()
	{
		const std::size_t X2=2, Y2=3, A=4, B=5;
		const RI::Tensor<Tdata> Tx = init_tensor<Tdata>({A,B,X2});
		const RI::Tensor<Tdata> Ty = init_tensor<Tdata>({A,B,Y2});
		const RI::Tensor<Tdata> Txy = RI::Tensor_Multiply::x2y2_abx2_aby2(Tx,Ty);
		FOR_0(x2,X2)
			FOR_0(y2,Y2)
			{
				Tdata result = 0;
				FOR_0(a,A)
					FOR_0(b,B)
						result += Tx(a,b,x2) * Ty(a,b,y2);
				assert(Txy(x2,y2) == result);
			}
	}

}

#undef FOR_0
