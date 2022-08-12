// ====================
//  Author: Peize Lin
//  Date:   2022.08.11
// ====================

#pragma once

#include <map>
#include <functional>

namespace Map_Operator
{
	template<typename Tkey, typename Tvalue, typename Tdata>
	std::map<Tkey,Tvalue> transform(
		const std::map<Tkey,Tvalue> &m_in,
		const std::function<Tdata(const Tdata&)> &func);
}

#include "Map_Operator-3.hpp"