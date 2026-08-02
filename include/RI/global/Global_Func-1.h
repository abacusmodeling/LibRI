// ===================
//  Author: Peize Lin
//  date: 2022.05.26
// ===================

#pragma once

#include "Tensor.h"

#include <vector>
#include <map>
#include <set>
#include <array>
#include <type_traits>

namespace RI
{

namespace Global_Func
{
	template<typename T> const T ZERO{};

	// tensor = find(m,i,j,k);
	//   <=>
	// tensor = m.at(i).at(j).at(k);
	// Peize Lin add 2022.05.26
	template<typename Tkey, typename Tdata,
		typename std::enable_if<std::is_arithmetic<Tdata>::value,bool>::type=0>
	inline const Tdata &find(
		const std::map<Tkey, Tdata> &m,
		const Tkey &key)
	{
		const auto &ptr = m.find(key);
		if(ptr==m.end())
			return ZERO<Tdata>;
		else
			return ptr->second;
	}
	template<typename Tkey, typename Tdata>
	inline const Tensor<Tdata> &find(
		const std::map<Tkey, Tensor<Tdata>> &m,
		const Tkey &key)
	{
		const auto &ptr = m.find(key);
		if(ptr==m.end())
			return ZERO<Tensor<Tdata>>;
		else
			return ptr->second;
	}
	template<typename Tkey, typename Tdata, std::size_t Ndim>
	inline const std::array<Tdata,Ndim> &find(
		const std::map<Tkey, std::array<Tdata,Ndim>> &m,
		const Tkey &key)
	{
		const auto &ptr = m.find(key);
		if(ptr==m.end())
			return ZERO<std::array<Tdata,Ndim>>;
		else
			return ptr->second;
	}
	template<typename Tkey, typename Tdata>
	inline const std::vector<Tdata> &find(
		const std::map<Tkey, std::vector<Tdata>> &m,
		const Tkey &key)
	{
		const auto &ptr = m.find(key);
		if(ptr==m.end())
			return ZERO<std::vector<Tdata>>;
		else
			return ptr->second;
	}
	template<typename Tkey0, typename Tkey1, typename Tvalue, typename... Tkeys>
	inline const auto &find(
		const std::map<Tkey0, std::map<Tkey1,Tvalue>> &m,
		const Tkey0 &key0,
		const Tkeys&... keys)
//	-> decltype(find( m.find(key)->second, keys... ))			// why error for C++ compiler high version
	{
		const auto &ptr = m.find(key0);
		if(ptr==m.end())
			return ZERO<typename std::remove_reference<decltype(find( ptr->second, keys... ))>::type>;
		else
			return find( ptr->second, keys... );
	}

	// These functions are designed to replace Global_Func::find,
	// the find result can be arithmetic type, Tensor, vector, array, or map.

	// Helper: compute return type of find_map with N keys (Map::mapped_type applied N times)
	template<typename Map, typename... Keys>
	struct find_map_result;

	template<typename Map, typename Key>
	struct find_map_result<Map, Key> {
		using type = typename Map::mapped_type;
	};

	template<typename Map, typename Key0, typename... Keys>
	struct find_map_result<Map, Key0, Keys...> {
		using type = typename find_map_result<typename Map::mapped_type, Keys...>::type;
	};

	// Base case: 1 key — returns const& to Map::mapped_type (ZERO sentinel if key not found)
	template<class Map, class Key>
	static inline const typename Map::mapped_type& find_map(const Map& map, const Key& key)
	{
		const auto& it = map.find(key);
		if (it != map.end()) return it->second;
		return ZERO<typename Map::mapped_type>;
	}

	// Recursive case: 2+ keys — peels off key0, recurses on remaining keys
	template<class Map, class Key0, class... Keys>
	static inline const typename find_map_result<Map, Key0, Keys...>::type&
	find_map(const Map& map, const Key0& key0, const Keys&... keys)
	{
		const auto it = map.find(key0);
		if (it != map.end())
			return find_map(it->second, keys...);
		return ZERO<typename find_map_result<Map, Key0, Keys...>::type>;
	}

	// in_set(3, {2,3,5,7})
	// Peize Lin add 2022.05.26
	template<typename T>
	inline bool in_set(const T &item, const std::set<T> &s)
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
	inline std::set<T> to_set(const std::vector<T> &v)
	{
		return std::set<T>(v.begin(), v.end());
	}
	template<typename T, std::size_t N>
	inline std::vector<T> to_vector(const std::array<T,N> &v)
	{
		return std::vector<T>(v.begin(), v.end());
	}

	/// @brief sorted unique union of two containers
	/// @tparam C1,C2  containers supporting .begin()/.end() and iterator-pair construction
	///               (e.g., std::vector, std::deque, std::list)
	template<typename C1, typename C2>
	static auto set_union(const C1& c1, const C2& c2)
		-> C1
	{
		static_assert(
			std::is_same<typename C1::value_type, typename C2::value_type>::value,
			"set_union: both containers must have the same value_type");
		std::set<typename C1::value_type> s(c1.begin(), c1.end());
		s.insert(c2.begin(), c2.end());
		return C1(s.begin(), s.end());
	}
}

}