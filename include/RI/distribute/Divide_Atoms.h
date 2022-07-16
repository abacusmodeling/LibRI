// ===================
//  Author: Peize Lin
//  date: 2022.07.13
// ===================

#pragma once

#include <vector>
#include <array>
#include <utility>

namespace Divide_Atoms
{
	// equally divide atoms:
	// 	[0,1,2]  [3,4,5]  [6,7]
	template<typename TA>
	std::vector<TA> divide_atoms(
		const int group_rank,
		const int group_size,
		const std::vector<TA> &atoms);

	// equally divide atoms:
	// 	[0,1]  [2]
	// with all period
	// 	[{0,0},{0,1},{1,0},{1,1}]  [{2,0},{2,1}]
    template<typename TA, typename Tperiod, size_t Ndim_period>
	std::vector<std::pair<TA,std::array<Tperiod,Ndim_period>>> divide_atoms(
		const int group_rank,
		const int group_size,
		const std::vector<TA> &atoms,
		const std::array<Tperiod,Ndim_period> &period);

	// equally divide atoms and periods:
	// 	[{0,0},{0,1},{1,0}]  [{1,1},{2,0},{2,1}]  [{3,0},{3,1}]
	template<typename TA, typename Tperiod, size_t Ndim_period>
	std::vector<std::pair<TA,std::array<Tperiod,Ndim_period>>> divide_atoms_periods(
		const int group_rank,
		const int group_size,
		const std::vector<TA> &atoms,
		const std::array<Tperiod,Ndim_period> &period);
}

#include "Divide_Atoms.hpp"