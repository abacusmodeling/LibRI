// ====================
//  Author: Peize Lin
//  Date:   2022.07.27
// ====================

#pragma once

#include <map>
#include <functional>

namespace Map_Operator
{
	template<typename Tkey, typename Tvalue, typename Tdata>
	std::map<Tkey,Tvalue> zip_union(
		const std::map<Tkey,Tvalue> &m1,
		const std::map<Tkey,Tvalue> &m2,
		const std::function<Tdata(const Tdata&,const Tdata&)> &func);
		
	template<typename Tkey, typename Tvalue, typename Tdata>
	std::map<Tkey,Tvalue> zip_intersection(
		const std::map<Tkey,Tvalue> &m1,
		const std::map<Tkey,Tvalue> &m2,
		const std::function<Tdata(const Tdata&,const Tdata&)> &func);
}

#include "Map_Operator-2.hpp"