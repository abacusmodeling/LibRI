// ===================
//  Author: Peize Lin
//  date: 2022.07.13
// ===================

#pragma once

#include "Divide_Atoms.h"
#include "global/Global_Func-3.h"

#include <numeric>
#include <stdexcept>
#include <string>

namespace Divide_Atoms
{
	/*
	traversal_period
		In: [2,3]
		Out: [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2]]
	*/
	template<typename Tperiod>
	std::vector<std::array<Tperiod, 1>> traversal_period(const std::array<Tperiod,1> &period)
	{
		std::vector<std::array<Tperiod, 1>> cells; 
		cells.reserve(std::accumulate( period.begin(), period.end(), 1, std::multiplies<Tperiod>() ));
		for(int item=0; item<period[0]; ++item)
			cells.push_back(std::array<Tperiod,1>{item});
		return cells;
	}

	template<typename Tperiod, size_t Ndim_period>
	std::vector<std::array<Tperiod, Ndim_period>> traversal_period(const std::array<Tperiod,Ndim_period> &period)
	{
		auto cat_array = [](
			const Tperiod &item_first,
			const std::array<Tperiod,Ndim_period-1> &array_latter)
		-> std::array<Tperiod,Ndim_period>
		{
			std::array<Tperiod,Ndim_period> array_all;
			array_all[0] = item_first;
			for(int i=1; i<Ndim_period; ++i)
				array_all[i] = array_latter[i-1];
			return array_all;
		};

		auto sub_array = [](
			const std::array<Tperiod,Ndim_period> &array_all)
		-> std::array<Tperiod,Ndim_period-1>
		{
			std::array<Tperiod,Ndim_period-1> array_sub;
			for(int i=1; i<Ndim_period; ++i)
				array_sub[i-1] = array_all[i];
			return array_sub;
		};

		std::vector<std::array<Tperiod, Ndim_period>> cells; 
		cells.reserve(std::accumulate( period.begin(), period.end(), 1, std::multiplies<Tperiod>() ));
		const std::array<Tperiod,Ndim_period-1> period_sub = sub_array(period);
		const std::vector<std::array<Tperiod, Ndim_period-1>> cells_sub = traversal_period(period_sub); 
		for(int item_first=0; item_first<period[0]; ++item_first)
			for(const auto &cell_sub : cells_sub)
				cells.push_back(cat_array(item_first, cell_sub));
		return cells;
	}

	template<typename TA>
	std::vector<TA> divide_atoms(
		const int group_rank,
		const int group_size,
		const std::vector<TA> &atoms)
	{
		const int mod = atoms.size() % group_size;
		const int index_begin = 
			(group_rank < mod)
			? group_rank * (atoms.size()/group_size+1)
			: mod * (atoms.size()/group_size+1) + (group_rank-mod) * (atoms.size()/group_size);
		const int index_size = 
			(group_rank < mod)
			? atoms.size()/group_size+1
			: atoms.size()/group_size;
		return std::vector<TA>(atoms.begin()+index_begin, atoms.begin()+index_begin+index_size);
	}

	template<typename TA, typename Tperiod, size_t Ndim_period>
	std::vector<std::pair<TA,std::array<Tperiod,Ndim_period>>> divide_atoms(
		const int group_rank,
		const int group_size,
		const std::vector<TA> &atoms,
		const std::array<Tperiod,Ndim_period> &period)
	{
		using TAp = std::pair<TA,std::array<Tperiod,Ndim_period>>;
		const std::vector<TA> atoms_divide = divide_atoms(group_rank, group_size, atoms);
		const std::vector<std::array<Tperiod, Ndim_period>> cells_origin = traversal_period(period);
		const std::vector<std::array<Tperiod, Ndim_period>> cells = Global_Func::mod_period(cells_origin, period);
		std::vector<TAp> atoms_periods_divide;
		atoms_periods_divide.reserve( atoms_divide.size() * cells.size() );
		for(const TA &atom : atoms_divide)
			for(const std::array<Tperiod,Ndim_period> &cell : cells)
				atoms_periods_divide.push_back(std::make_pair(atom,cell));
		return atoms_periods_divide;
	}

	template<typename TA, typename Tperiod, size_t Ndim_period>
	std::vector<std::pair<TA,std::array<Tperiod,Ndim_period>>> divide_atoms_periods(
		const int group_rank,
		const int group_size,
		const std::vector<TA> &atoms,
		const std::array<Tperiod,Ndim_period> &period)
	{
		const std::vector<std::array<Tperiod, Ndim_period>> cells_origin = traversal_period(period);
		const std::vector<std::array<Tperiod, Ndim_period>> cells = Global_Func::mod_period(cells_origin, period);
		const int mod = atoms.size() * cells.size() % group_size;
		const int index_begin = 
			(group_rank < mod)
			? group_rank * (atoms.size()*cells.size()/group_size+1)
			: mod * (atoms.size()*cells.size()/group_size+1) + (group_rank-mod) * (atoms.size()*cells.size()/group_size);
		const int index_size = 
			(group_rank < mod)
			? atoms.size()*cells.size()/group_size+1
			: atoms.size()*cells.size()/group_size;
		const int index_end = index_begin + index_size;
		
		std::vector<std::pair<TA,std::array<Tperiod,Ndim_period>>> atoms_periods_divide;
		atoms_periods_divide.reserve(index_size);

		const int iatom_begin = index_begin / cells.size();
		int index = iatom_begin * cells.size();
		for(int iatom=iatom_begin; iatom<atoms.size(); ++iatom)
		{
			for(const std::array<Tperiod,Ndim_period> &cell : cells)
			{
				if(index>=index_begin)
					atoms_periods_divide.push_back(std::make_pair(atoms[iatom], cell));
				++index;
				if(index>=index_end)
					return atoms_periods_divide;
			}
		}
		throw std::range_error(std::string(__FILE__)+" line "+std::to_string(__LINE__));
	}
}