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
	Tensor<Tdata> x0y2_x0ab_aby2(const Tensor<Tdata> &Tx, const Tensor<Tdata> &Ty)
	{
		assert(Tx.shape.size()==3);
		assert(Ty.shape.size()==3);
		Tensor<Tdata> Txy({Tx.shape[0], Ty.shape[2]});
		Blas_Interface::gemm(
			'N', 'N',
			Tx.shape[0],
			Ty.shape[2],
			Tx.shape[1] * Tx.shape[2],
			Tdata(1.0), Tx.ptr(), Ty.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}

	// Txy(x2,y0) = Tx(a,b,x2) * Ty(y0,a,b)
	template<typename Tdata>
	Tensor<Tdata> x2y0_abx2_y0ab(const Tensor<Tdata> &Tx, const Tensor<Tdata> &Ty)
	{
		assert(Tx.shape.size()==3);
		assert(Ty.shape.size()==3);
		Tensor<Tdata> Txy({Tx.shape[2], Ty.shape[0]});
		Blas_Interface::gemm(
			'T', 'T',
			Tx.shape[2],
			Ty.shape[0],
			Tx.shape[0] * Tx.shape[1],
			Tdata(1.0), Tx.ptr(), Ty.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}

}

}
