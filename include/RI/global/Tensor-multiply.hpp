// ===================
//  Author: Peize Lin
//  date: 2021.10.31
// ===================

#pragma once

#include "Tensor.h"

#include "global/Blas_Interface-Tensor.h"
#include <cassert>
#include <stdexcept>
#include <string>

template<typename T>
Tensor<T> operator* (const Tensor<T> &t1, const Tensor<T> &t2)
{
	switch(t1.shape.size())
	{
		case 1:
		{
			switch(t2.shape.size())
			{
				case 1:
				{
					assert(t1.shape[0] == t2.shape[0]);
					Tensor<T> t({1});
					t(0) = Blas_Interface::dot(t1, t2);
					return t;
				}
				case 2:
				{
					assert(t1.shape[0] == t2.shape[0]);
					return Blas_Interface::gemv('T', 1.0, t2, t1);
				}
				default:;
					throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
			}
		}
		case 2:
		{
			switch(t2.shape.size())
			{
				case 1:
				{
					assert(t1.shape[1] == t2.shape[0]);
					return Blas_Interface::gemv('N', 1.0, t1, t2);
				}
				case 2:
				{
					assert(t1.shape[1] == t2.shape[0]);
					return Blas_Interface::gemm('N', 'N', 1.0, t1, t2);
				}
				default:
					throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
			}
		}
		default:
			throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
	}
}