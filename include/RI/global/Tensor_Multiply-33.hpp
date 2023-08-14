// ===================
//  Author: Peize Lin
//  date: 2023.08.03
// ===================

#pragma once

#include "Tensor_Multiply.h"

namespace RI
{

namespace Tensor_Multiply
{
	// Txy(x0,y2) = Tx(x0,a,b) * Ty(a,b,y2)
	template<typename Tdata>
	Tensor<Tdata> x0y2_x1y0_x2y1(const Tensor<Tdata> &Tx, const Tensor<Tdata> &Ty)
	{
		assert(Tx.shape.size()==3);
		assert(Ty.shape.size()==3);
		assert(Tx.shape[1]==Ty.shape[0]);
		assert(Tx.shape[2]==Ty.shape[1]);
		const std::size_t x12 = Tx.shape[1] * Tx.shape[2];
		Tensor<Tdata> Txy({Tx.shape[0], Ty.shape[2]});
		Blas_Interface::gemm(
			'N', 'N', Tx.shape[0], Ty.shape[2], x12,
			Tdata(1.0), Tx.ptr(), Ty.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}

	// Txy(x2,y0) = Tx(a,b,x2) * Ty(y0,a,b)
	template<typename Tdata>
	Tensor<Tdata> x2y0_x0y1_x1y2(const Tensor<Tdata> &Tx, const Tensor<Tdata> &Ty)
	{
		assert(Tx.shape.size()==3);
		assert(Ty.shape.size()==3);
		assert(Tx.shape[0]==Ty.shape[1]);
		assert(Tx.shape[1]==Ty.shape[2]);
		const std::size_t x01 = Tx.shape[0] * Tx.shape[1];
		Tensor<Tdata> Txy({Tx.shape[2], Ty.shape[0]});
		Blas_Interface::gemm(
			'T', 'T', Tx.shape[2], Ty.shape[0], x01,
			Tdata(1.0), Tx.ptr(), Ty.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}
}

}