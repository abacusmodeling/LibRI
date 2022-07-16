// ===================
//  Author: Peize Lin
//  date: 2022.07.13
// ===================

#pragma once

#include "Array_Operator.h"
#include <vector>
#include <array>

namespace Global_Func
{
	template<typename Tperiod, size_t Ndim_period>
	std::vector<std::array<Tperiod, Ndim_period>> mod_period(
		const std::vector<std::array<Tperiod, Ndim_period>> &cells_origin,
		const std::array<Tperiod,Ndim_period> &period)
	{
		using namespace Array_Operator;
		std::vector<std::array<Tperiod, Ndim_period>> cells_mod(cells_origin.size());
		for(int i=0; i<cells_origin.size(); ++i)
			cells_mod[i] = cells_origin[i] % period;
		return cells_mod;
	}
}