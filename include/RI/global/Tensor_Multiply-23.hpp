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
	// Txy(x1,y0,y1) = Tx(a,x1) * Ty(y0,y1,a)
	template<typename Tdata>
	Tensor<Tdata> x1y0y1_ax1_y0y1a(const Tensor<Tdata> &Tx, const Tensor<Tdata> &Ty)
	{
		assert(Tx.shape.size()==2);
		assert(Ty.shape.size()==3);
		Tensor<Tdata> Txy({Tx.shape[1], Ty.shape[0], Ty.shape[1]});
		Blas_Interface::gemm(
			'T', 'T',
			Tx.shape[1],
			Ty.shape[0] * Ty.shape[1],
			Tx.shape[0],
			Tdata(1.0), Tx.ptr(), Ty.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}

}

}
