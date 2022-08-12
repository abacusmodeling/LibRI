// ===================
//  Author: Peize Lin
//  date: 2021.10.31
// ===================

#pragma once

#include "Tensor.h"
#include "Blas_Interface-Tensor.h"

#include <complex>
#include <memory>
#include <vector>
#include <numeric>
#include <functional>
#include <cassert>
#include <limits>

template<typename T>
Tensor<T>::Tensor (const std::vector<size_t> &shape_in)
{
	this->shape = shape_in;
	const size_t shape_all = get_shape_all();
	this->data = std::make_shared<std::valarray<T>>(0,shape_all);
}

template<typename T>
Tensor<T>::Tensor (const std::vector<size_t> &shape_in, std::shared_ptr<std::valarray<T>> data_in)
{
	assert( std::accumulate(shape_in.begin(), shape_in.end(), static_cast<size_t>(1), std::multiplies<size_t>() ) == data_in->size() );
	this->shape = shape_in;
	this->data = data_in;
}

/*
template<typename T>
Tensor<T>::Tensor (const Tensor<T> &t_in)
{
	this->shape = t_in.shape;
	this->data = t_in.data;
}

template<typename T>
Tensor<T>::Tensor (Tensor<T> &&t_in)
{
	this->shape = std::move(t_in.shape);
	this->data = std::move(t_in.data);
}

template<typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor<T> &t_in)
{
	this->shape = t_in.shape;
	this->data = t_in.data;
	return *this;
}

template<typename T>
Tensor<T> &Tensor<T>::operator=(Tensor<T> &&t_in)
{
	this->shape = std::move(t_in.shape);
	this->data = std::move(t_in.data);
	return *this;
}
*/

template<typename T>
size_t Tensor<T>::get_shape_all() const
{
	return std::accumulate(this->shape.begin(), this->shape.end(), static_cast<size_t>(1), std::multiplies<size_t>() );
}

template<typename T>
Tensor<T> Tensor<T>::reshape (const std::vector<size_t> &shape_in) const
{
	assert(
		std::accumulate(shape_in.begin(), shape_in.end(), static_cast<size_t>(1), std::multiplies<size_t>())
		== this->get_shape_all() );
	return Tensor<T>(shape_in, this->data);
}

template<typename T>
Tensor<T> Tensor<T>::copy() const
{
	Tensor<T> t(this->shape);
	*t.data = *this->data;
	return t;
}


template<typename T>
T& Tensor<T>::operator() (const size_t i0) const
{
	assert(this->shape.size()==1);
	assert(i0>=0);	assert(i0<this->shape[0]);
	return (*this->data)[i0];
}
template<typename T>
T& Tensor<T>::operator() (const size_t i0, const size_t i1) const
{
	assert(this->shape.size()==2);
	assert(i0>=0);	assert(i0<this->shape[0]);
	assert(i1>=0);	assert(i1<this->shape[1]);
	return (*this->data)[i0*this->shape[1]+i1];
}
template<typename T>
T& Tensor<T>::operator() (const size_t i0, const size_t i1, const size_t i2) const
{
	assert(this->shape.size()==3);
	assert(i0>=0);	assert(i0<this->shape[0]);
	assert(i1>=0);	assert(i1<this->shape[1]);
	assert(i2>=0);	assert(i2<this->shape[2]);
	return (*this->data)[(i0*this->shape[1]+i1)*this->shape[2]+i2];
}

template<typename T1, typename T2>
bool same_shape (const Tensor<T1> &t1, const Tensor<T2> &t2)
{
	if(t1.shape.size() != t2.shape.size())
		return false;
	for(size_t ishape=0; ishape<t1.shape.size(); ++ishape)
		if(t1.shape[ishape] != t2.shape[ishape])
			return false;
	return true;	
}


template<typename T>
Tensor<T> operator+ (const Tensor<T> &t1, const Tensor<T> &t2)
{
	assert(same_shape(t1,t2));
	Tensor<T> t(t1.shape);
	*t.data = *t1.data + *t2.data;
	return t;
}
template<typename T>
Tensor<T> operator- (const Tensor<T> &t1, const Tensor<T> &t2)
{
	assert(same_shape(t1,t2));
	Tensor<T> t(t1.shape);
	*t.data = *t1.data - *t2.data;
	return t;
}

template<typename T>
Tensor<T> operator* (const T &t1, const Tensor<T> &t2)
{
	Tensor<T> t(t2.shape);
	*t.data = t1 * *t2.data;
	return t;
}

template<typename T>
Tensor<T> operator* (const Tensor<T> &t1, const T &t2)
{
	Tensor<T> t(t1.shape);
	*t.data = *t1.data * t2;
	return t;
}

/*
template<typename T>
Tensor<T> &Tensor<T>::operator+= (const Tensor &t)
{
	assert(same_shape(*this,t));
	*this->data += *t.data;
	return *this;
}
*/

template<typename T>
Tensor<T> Tensor<T>::transpose() const
{
	assert(this->shape.size()==2);
	Tensor<T> t({this->shape[1], this->shape[0]});
	for(size_t i0=0; i0<this->shape[0]; ++i0)
		for(size_t i1=0; i1<this->shape[1]; ++i1)
			t(i1,i0) = (*this)(i0,i1);
	return t;
}

template<typename T>
Global_Func::To_Real_t<T> Tensor<T>::norm(const double p) const
{
	using T_res = Global_Func::To_Real_t<T>;
	if(p==2)
		return Blas_Interface::nrm2(*this);
	else if(p==1)
	{
		T_res s = 0;
		for(size_t i=0; i<this->data->size(); ++i)
			s += std::abs((*this->data)[i]);
		return s;
	}
	else if(p==std::numeric_limits<double>::max())
	{
		T_res s = 0;
		for(size_t i=0; i<this->data->size(); ++i)
			s = std::max(std::real(s), std::abs((*this->data)[i]));
		return s;
	}
	else
	{
		T_res s = 0;
		for(size_t i=0; i<this->data->size(); ++i)
			s += std::pow(std::abs((*this->data)[i]), p);
		return std::pow(s,1.0/p);
	}
}


namespace Global_Func
{
	template<typename Tout, typename Tin>
	Tensor<Tout> convert(const Tensor<Tin> &t_in)
	{
		Tensor<Tout> t_out(t_in.shape);
		for(size_t i=0; i<t_out.data->size(); ++i)
			(*t_out.data)[i] = Global_Func::convert<Tout>((*t_in.data)[i]);
		return t_out;
	}
}