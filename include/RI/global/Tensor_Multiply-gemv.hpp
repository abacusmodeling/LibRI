#include "Tensor_Multiply.h"

namespace RI
{

namespace Tensor_Multiply
{
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
			for (int d = 0; d < dim; ++d)
				shape_vector.push_back(Tx.shape[d]);
		Tensor<Tdata> Txy(shape_vector);
		Blas_Interface::gemv(
			'N', Txy.get_shape_all(), ny,
			Tdata(1.0), Tx.ptr(), Vy.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}

	// Txy(x0,x1,..., xM) = Tx(y0, y1,y2, ..., yN, x0, x1, ..., xM) * Vy(y0, y1, ..., yN)
	template<typename Tdata>
	Tensor<Tdata> gemv_trans(const Tensor<Tdata>& Tx, const Tensor<Tdata>& Vy)
	{
		assert(Tx.shape.size() >= Vy.shape.size());
		const std::size_t ny = Vy.get_shape_all();
		assert(Tx.get_shape_all() % ny == 0);
		const std::size_t dim = Tx.shape.size() - Vy.shape.size();
		std::vector<std::size_t> shape_vector;
		if (dim == 0)
			shape_vector.push_back(1);
		else
			for (int d = 0; d < dim; ++d)
				shape_vector.push_back(Tx.shape[Vy.shape.size() + d]);
		Tensor<Tdata> Txy(shape_vector);
		Blas_Interface::gemv(
			'T', ny, Txy.get_shape_all(),
			Tdata(1.0), Tx.ptr(), Vy.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}
}
}