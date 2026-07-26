// ===================
//  Author: maki49
//  date: 2026.07.10
// ===================

#pragma once

#include "RI/distribute/Divide_Atoms_Weighted.h"
#include "unittests/print_stl.h"

namespace Divide_Atoms_Weighted_Test
{
	// 27 light atoms (weight=2) followed by 4 heavy ones (weight=25): sum(weight) = 154.
	// Balancing the atom count would give the last group 5 heavy-ish atoms, i.e. 4x the load
	// of the first. Balancing sum(weight) gives every group 25 or 26, at the price of
	// non-contiguous groups.
	static std::map<std::size_t,std::size_t> weight_heavy_tail()
	{
		std::map<std::size_t,std::size_t> weight;
		for(std::size_t i=0; i<27; ++i)	weight[i] = 2;
		for(std::size_t i=27; i<31; ++i)	weight[i] = 25;
		return weight;
	}

	static void test_divide_atoms_weight()
	{
		const std::size_t group_size = 6;
		std::vector<std::size_t> atoms(31);
		for(std::size_t i=0; i<atoms.size(); ++i)
			atoms[i]=i;
		const std::map<std::size_t,std::size_t> weight = weight_heavy_tail();
		for(std::size_t i=0; i<group_size; ++i)
			std::cout<<RI::Divide_Atoms_Weighted::divide_atoms(i, group_size, atoms, weight)<<std::endl;
	}
	/*
		26|	27|
		28|
		29|
		30|
		0|	2|	4|	6|	8|	10|	12|	14|	16|	18|	20|	22|	24|
		1|	3|	5|	7|	9|	11|	13|	15|	17|	19|	21|	23|	25|

		sum(weight):    27|	25|	25|	25|	26|	26|	  (total 154, ideal 25.67)
		atom count:   2|	 1|	 1|	 1|	13|	13|
		The atom counts are deliberately lopsided: each heavy atom alone is worth
		12 light ones, so equal load means unequal counts.
	*/


	static void test_divide_atoms_periods_weight()
	{
		const std::size_t group_size = 6;
		std::vector<std::size_t> atoms(31);
		for(std::size_t i=0; i<atoms.size(); ++i)
			atoms[i]=i;
		const std::array<int,1> period = {2};
		const std::map<std::size_t,std::size_t> weight = weight_heavy_tail();
		for(std::size_t i=0; i<group_size; ++i)
			std::cout<<RI::Divide_Atoms_Weighted::divide_atoms_periods(i, group_size, atoms, period, weight)<<std::endl;
	}
	/*
		Every {atom,cell} is one item of weight weight[atom]; sum over all 62 items = 308.
		Unlike divide_atoms above, an atom's two cells may land in different groups.

		{ 26, 0	 }|	{ 27, 0	 }|	{ 30, 0	 }|
		{ 26, -1	 }|	{ 27, -1	 }|	{ 30, -1	 }|
		{ 0, 0	 }|	{ 2, 0	 }|	{ 4, 0	 }|	{ 6, 0	 }|	{ 8, 0	 }|	{ 10, 0	 }|	{ 12, 0	 }|	{ 14, 0	 }|	{ 16, 0	 }|	{ 18, 0	 }|	{ 20, 0	 }|	{ 22, 0	 }|	{ 24, 0	 }|	{ 28, 0	 }|
		{ 0, -1	 }|	{ 2, -1	 }|	{ 4, -1	 }|	{ 6, -1	 }|	{ 8, -1	 }|	{ 10, -1	 }|	{ 12, -1	 }|	{ 14, -1	 }|	{ 16, -1	 }|	{ 18, -1	 }|	{ 20, -1	 }|	{ 22, -1	 }|	{ 24, -1	 }|	{ 28, -1	 }|
		{ 1, 0	 }|	{ 3, 0	 }|	{ 5, 0	 }|	{ 7, 0	 }|	{ 9, 0	 }|	{ 11, 0	 }|	{ 13, 0	 }|	{ 15, 0	 }|	{ 17, 0	 }|	{ 19, 0	 }|	{ 21, 0	 }|	{ 23, 0	 }|	{ 25, 0	 }|	{ 29, 0	 }|
		{ 1, -1	 }|	{ 3, -1	 }|	{ 5, -1	 }|	{ 7, -1	 }|	{ 9, -1	 }|	{ 11, -1	 }|	{ 13, -1	 }|	{ 15, -1	 }|	{ 17, -1	 }|	{ 19, -1	 }|	{ 21, -1	 }|	{ 23, -1	 }|	{ 25, -1	 }|	{ 29, -1	 }|

		sum(weight):	52|	52|	51|	51|	51|	51|	  (total 308, ideal 51.33)
	*/
}