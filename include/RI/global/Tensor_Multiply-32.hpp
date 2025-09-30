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
	// Txy(x0,x1,y0) = Tx(x0,x1,a) * Ty(y0,a)
	template<typename Tdata>
	Tensor<Tdata> x0x1y0_x0x1a_y0a(const Tensor<Tdata> &Tx, const Tensor<Tdata> &Ty)
	{
		assert(Tx.shape.size()==3);
		assert(Ty.shape.size()==2);
		Tensor<Tdata> Txy({Tx.shape[0], Tx.shape[1], Ty.shape[0]});
		Blas_Interface::gemm(
			'N', 'T',
			Tx.shape[0] * Tx.shape[1],
			Ty.shape[0],
			Tx.shape[2],
			Tdata(1.0), Tx.ptr(), Ty.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}

	// Txy(x0,x1,y1) = Tx(x0,x1,a) * Ty(a,y1)
	template<typename Tdata>
	Tensor<Tdata> x0x1y1_x0x1a_ay1(const Tensor<Tdata> &Tx, const Tensor<Tdata> &Ty)
	{
		assert(Tx.shape.size()==3);
		assert(Ty.shape.size()==2);
		Tensor<Tdata> Txy({Tx.shape[0], Tx.shape[1], Ty.shape[1]});
		Blas_Interface::gemm(
			'N', 'N',
			Tx.shape[0] * Tx.shape[1],
			Ty.shape[1],
			Tx.shape[2],
			Tdata(1.0), Tx.ptr(), Ty.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}

	// Txy(x1,x2,y0) = Tx(a,x1,x2) * Ty(y0,a)
	template<typename Tdata>
	Tensor<Tdata> x1x2y0_ax1x2_y0a(const Tensor<Tdata> &Tx, const Tensor<Tdata> &Ty)
	{
		assert(Tx.shape.size()==3);
		assert(Ty.shape.size()==2);
		Tensor<Tdata> Txy({Tx.shape[1], Tx.shape[2], Ty.shape[0]});
		Blas_Interface::gemm(
			'T', 'T',
			Tx.shape[1] * Tx.shape[2],
			Ty.shape[0],
			Tx.shape[0],
			Tdata(1.0), Tx.ptr(), Ty.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}

	// Txy(x1,x2,y1) = Tx(a,x1,x2) * Ty(a,y1)
	template<typename Tdata>
	Tensor<Tdata> x1x2y1_ax1x2_ay1(const Tensor<Tdata> &Tx, const Tensor<Tdata> &Ty)
	{
		assert(Tx.shape.size()==3);
		assert(Ty.shape.size()==2);
		Tensor<Tdata> Txy({Tx.shape[1], Tx.shape[2], Ty.shape[1]});
		Blas_Interface::gemm(
			'T', 'N',
			Tx.shape[1] * Tx.shape[2],
			Ty.shape[1],
			Tx.shape[0],
			Tdata(1.0), Tx.ptr(), Ty.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}

	// Txy(x0,x1,..., xM) = Tx(x0, x1,x2, ..., xM, y0, y1, ..., yN) * Vy(y0, y1, ..., yN)
	template<typename Tdata>
	Tensor<Tdata> gemv(const Tensor<Tdata>& Tx, const Tensor<Tdata>& Vy)
	{
		assert(Tx.shape.size() >= Vy.shape.size());
		const std::size_t ny = Vy.get_shape_all();
		assert(Tx.get_shape_all() % ny == 0);
		const std::size_t dim = Tx.shape.size() - Vy.shape.size();
		std::vector<std::size_t> shape_vector;
		if (dim == 0)
			shape_vector.push_back(1);
		else
			for (int d = 0; d < dim;++d)
				shape_vector.push_back(Tx.shape[d]);
		Tensor<Tdata> Txy(shape_vector);
		Blas_Interface::gemv(
			'N', Txy.get_shape_all(), ny,
			Tdata(1.0), Tx.ptr(), ny, Vy.ptr(), 1,
			Tdata(0.0), Txy.ptr(), 1);
		return Txy;
	}

}

}
