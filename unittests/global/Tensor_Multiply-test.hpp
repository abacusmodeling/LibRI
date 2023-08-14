// ===================
//  Author: Peize Lin
//  date: 2023.08.03
// ===================

#pragma once

#include "RI/global/Tensor_Multiply.h"

#define FOR_0(i,N)	\
	for(std::size_t i=0; i<N; ++i)

namespace Tensor_Multiply_Test
{
	template<typename Tdata>
	RI::Tensor<Tdata> init_tensor(const RI::Shape_Vector &shape)
	{
		RI::Tensor<Tdata> T(shape);
		const std::size_t size = T.get_shape_all();
		for(std::size_t i=0; i<size; ++i)
			T.ptr()[i] = i;
		return T;
	}



	template<typename Tdata>
	void main23()
	{
		// Txy(x1,y0,y1) = Tx(a,x1) * Ty(y0,y1,a)
		{
			const std::size_t A=5, X1=2, Y0=3, Y1=4;
			const RI::Tensor<Tdata> Tx = init_tensor<Tdata>({A,X1});
			const RI::Tensor<Tdata> Ty = init_tensor<Tdata>({Y0,Y1,A});
			const RI::Tensor<Tdata> Txy = RI::Tensor_Multiply::x1y0y1_x0y2(Tx, Ty);
			FOR_0(x1,X1)
				FOR_0(y0,Y0)
					FOR_0(y1,Y1)
					{
						Tdata result = 0;
						FOR_0(a,A)
							result += Tx(a,x1) * Ty(y0,y1,a);
						assert(Txy(x1,y0,y1) == result);
					}
		}
	}

	template<typename Tdata>
	void main32()
	{
		// Txy(x1,x2,y1) = Tx(a,x1,x2) * Ty(a,y1)
		{
			const std::size_t A=5, X1=2, X2=3, Y1=4;
			const RI::Tensor<Tdata> Tx = init_tensor<Tdata>({A,X1,X2});
			const RI::Tensor<Tdata> Ty = init_tensor<Tdata>({A,Y1});
			const RI::Tensor<Tdata> Txy = RI::Tensor_Multiply::x1x2y1_x0y0(Tx, Ty);
			FOR_0(x1,X1)
				FOR_0(x2,X2)
					FOR_0(y1,Y1)
					{
						Tdata result = 0;
						FOR_0(a,A)
							result += Tx(a,x1,x2) * Ty(a,y1);
						assert(Txy(x1,x2,y1) == result);
					}
		}

		// Txy(x0,x1,y0) = T(x0,x1,a) * T(y0,a)
		{
			const std::size_t A=5, X0=2, X1=3, Y0=4;
			const RI::Tensor<Tdata> Tx = init_tensor<Tdata>({X0,X1,A});
			const RI::Tensor<Tdata> Ty = init_tensor<Tdata>({Y0, A});
			const RI::Tensor<Tdata> Txy = RI::Tensor_Multiply::x0x1y0_x2y1(Tx,Ty);
			FOR_0(x0,X0)
				FOR_0(x1,X1)
					FOR_0(y0,Y0)
					{
						Tdata result = 0;
						FOR_0(a,A)
							result += Tx(x0,x1,a) * Ty(y0,a);
						assert(Txy(x0,x1,y0) == result);
					}
		}

		// Txy(x1,x2,y0) = Tx(a,x1,x2) * Ty(y0,a)
		{
			const std::size_t A=5, X1=2, X2=3, Y0=4;
			const RI::Tensor<Tdata> Tx = init_tensor<Tdata>({A,X1,X2});
			const RI::Tensor<Tdata> Ty = init_tensor<Tdata>({Y0, A});
			const RI::Tensor<Tdata> Txy = RI::Tensor_Multiply::x1x2y0_x0y1(Tx,Ty);
			FOR_0(x1,X1)
				FOR_0(x2,X2)
					FOR_0(y0,Y0)
					{
						Tdata result = 0;
						FOR_0(a,A)
							result += Tx(a,x1,x2) * Ty(y0,a);
						assert(Txy(x1,x2,y0) == result);
					}
		}

		// Txy(x0,x1,y1) = Tx(x0,x1,a) * Ty(a,y1)
		{
			const std::size_t A=5, X0=2, X1=3, Y1=4;
			const RI::Tensor<Tdata> Tx = init_tensor<Tdata>({X0,X1,A});
			const RI::Tensor<Tdata> Ty = init_tensor<Tdata>({A, Y1});
			const RI::Tensor<Tdata> Txy = RI::Tensor_Multiply::x0x1y1_x2y0(Tx,Ty);
			FOR_0(x0,X0)
				FOR_0(x1,X1)
					FOR_0(y1,Y1)
					{
						Tdata result = 0;
						FOR_0(a,A)
							result += Tx(x0,x1,a) * Ty(a,y1);
						assert(Txy(x0,x1,y1) == result);
					}
		}
	}



	template<typename Tdata>
	void main33()
	{
		// Txy(x0,y2) = Tx(x0,a,b) * Ty(a,b,y2)
		{
			const std::size_t A=5, B=4, X0=3, Y2=2;
			const RI::Tensor<Tdata> Tx = init_tensor<Tdata>(RI::Shape_Vector({X0,A,B}));
			const RI::Tensor<Tdata> Ty = init_tensor<Tdata>(RI::Shape_Vector({A,B,Y2}));
			const RI::Tensor<Tdata> Txy = RI::Tensor_Multiply::x0y2_x1y0_x2y1(Tx, Ty);
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
		{
			const std::size_t A=5, B=4, X2=3, Y0=2;
			const RI::Tensor<Tdata> Tx = init_tensor<Tdata>(RI::Shape_Vector({A,B,X2}));
			const RI::Tensor<Tdata> Ty = init_tensor<Tdata>(RI::Shape_Vector({Y0,A,B}));
			const RI::Tensor<Tdata> Txy = RI::Tensor_Multiply::x2y0_x0y1_x1y2(Tx, Ty);
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
	}


	
	template<typename Tdata>
	void main()
	{
		main23<Tdata>();
		main32<Tdata>();
		main33<Tdata>();
	}
}

#undef FOR_0