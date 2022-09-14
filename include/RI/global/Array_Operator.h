// ===================
//  Author: Peize Lin
//  date: 2022.02.18
// ===================

#pragma once

#include <array>

namespace Array_Operator
{
	template<typename T, std::size_t N>
	std::array<T,N> operator%(const std::array<T,N> &v1, const std::array<T,N> &v2)
	{
		auto mod = [](const int i, const int n){ return (i%n+3*n/2)%n-n/2; };			// [-n/2,n/2]
	//	auto mod = [](const int i, const int n){ return (i%n+n)%n; };					// [0,n)
	//	auto mod = [](const int i, const int n){ return i%n; };
		std::array<T,N> v;
		for(std::size_t i=0; i<N; ++i)
			v[i] = mod(v1[i], v2[i]);
		return v;
	}

	template<typename T, std::size_t N>
	std::array<T,N> operator+(const std::array<T,N> &v1, const std::array<T,N> &v2)
	{
		std::array<T,N> v;
		for(std::size_t i=0; i<N; ++i)
			v[i] = v1[i] + v2[i];
		return v;
	}

	template<typename T, std::size_t N>
	std::array<T,N> operator-(const std::array<T,N> &v1, const std::array<T,N> &v2)
	{
		std::array<T,N> v;
		for(std::size_t i=0; i<N; ++i)
			v[i] = v1[i] - v2[i];
		return v;
	}

	template<typename T, std::size_t N>
	std::array<T,N> operator-(const std::array<T,N> &v_in)
	{
		std::array<T,N> v;
		for(std::size_t i=0; i<N; ++i)
			v[i] = -v_in[i];
		return v;
	}

}