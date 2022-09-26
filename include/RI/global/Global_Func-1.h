// ===================
//  Author: Peize Lin
//  date: 2022.05.26
// ===================

#pragma once

#include "Tensor.h"

#include <map>
#include <set>
#include <type_traits>

namespace Global_Func
{
	// tensor = find(m,i,j,k);
	//   <=>
	// tensor = m.at(i).at(j).at(k);
	// Peize Lin add 2022.05.26
	template<typename Tkey, typename Tdata,
		typename std::enable_if<std::is_arithmetic<Tdata>::value,bool>::type=0>
	Tdata find(
		const std::map<Tkey, Tdata> &m,
		const Tkey key)
	{
		const auto ptr = m.find(key);
		if(ptr==m.end())
			return 0;
		else
			return ptr->second;
	}
	template<typename Tkey, typename Tdata>
	Tensor<Tdata> find(
		const std::map<Tkey, Tensor<Tdata>> &m,
		const Tkey key)
	{
		const auto ptr = m.find(key);
		if(ptr==m.end())
			return Tensor<Tdata>{};
		else
			return ptr->second;
	}
	template<typename Tkey, typename Tvalue, typename... Tkeys>
	auto find(
		const std::map<Tkey, Tvalue> &m,
		const Tkey key,
		const Tkeys... keys)
	-> decltype(find( m.find(key)->second, keys... ))
	{
		const auto ptr = m.find(key);
		if(ptr==m.end())
			return decltype(find( ptr->second, keys... )){};
		else
			return find( ptr->second, keys... );
	}

	// in_set(3, {2,3,5,7})
	// Peize Lin add 2022.05.26
	template<typename T>
	bool in_set(const T &item, const std::set<T> &s)
	{
		return s.find(item) != s.end();
	}

	template<typename Tkey, typename Tvalue>
	std::vector<Tkey> map_key_to_vec(const std::map<Tkey,Tvalue> &m)
	{
		std::vector<Tkey> v;
		v.reserve(m.size());
		for(const auto &im : m)
			v.push_back(im.first);
		return v;
	}

	template<typename Tkey, typename Tvalue>
	std::vector<Tkey> map_value_to_vec(const std::map<Tkey,Tvalue> &m)
	{
		std::vector<Tvalue> v;
		v.reserve(m.size());
		for(const auto &im : m)
			v.push_back(im.second);
		return v;
	}

	template<typename T>
	std::set<T> to_set(const std::vector<T> &v)
	{
		return std::set<T>(v.begin(), v.end());
	}
	template<typename T, std::size_t N>
	std::vector<T> to_vector(const std::array<T,N> &v)
	{
		return std::vector<T>(v.begin(), v.end());
	}
}