// ===================
//  Author: Peize Lin
//  date: 2021.10.31
// ===================

#pragma once

#include "../global/Global_Func-2.h"
#include <memory>
#include <vector>
#include <valarray>
#include <initializer_list>
#include <cassert>

// Attention: Unless using t2=t1.copy() explicitly, t2 and t1 share the same memory for all t2=t1, t2(t1), ...

template<typename T>
class Tensor
{
public:
	std::vector<size_t> shape;
	std::shared_ptr<std::valarray<T>> data;
	
	Tensor(){};
	explicit Tensor (const std::vector<size_t> &shape_in);
	explicit Tensor (const std::vector<size_t> &shape_in, std::shared_ptr<std::valarray<T>> data_in);

	inline size_t get_shape_all() const;
	inline Tensor reshape (const std::vector<size_t> &shape_in) const;
	
	Tensor copy() const;

	inline T& operator() (const size_t i0) const;
	inline T& operator() (const size_t i0, const size_t i1) const;
	inline T& operator() (const size_t i0, const size_t i1, const size_t i2) const;

	Tensor transpose() const;

	// ||d||_p = (|d_1|^p+|d_2|^p+...)^{1/p}
	// if(p==std::numeric_limits<double>::max())    ||d||_max = max_i |d_i|
	Global_Func::To_Real_t<T> norm(const double p) const;

	T* ptr()const{ return &(*this->data)[0]; }

	bool empty() const { return shape.empty(); }
	operator bool () const { return !shape.empty(); }

	Tensor & operator += (const Tensor &);

	template <class Archive> void serialize( Archive & ar ){ ar(shape); ar(data); }		// for cereal
};


template<typename T>
Tensor<T> operator+ (const Tensor<T> &t1, const Tensor<T> &t2);
template<typename T>
Tensor<T> operator- (const Tensor<T> &t1, const Tensor<T> &t2);

template<typename T>
Tensor<T> operator* (const Tensor<T> &t1, const Tensor<T> &t2);

#include "Blas_Interface-Tensor.h"