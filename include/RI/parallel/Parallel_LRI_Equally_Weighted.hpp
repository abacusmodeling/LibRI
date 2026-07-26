// ===================
//  Author: maki49
//  date: 2026.07.10
// ===================

#pragma once

#include "Parallel_LRI_Equally_Weighted.h"
#include "../distribute/Distribute_Equally_Weighted.h"

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void Parallel_LRI_Equally_Weighted<TA,Tcell,Ndim,Tdata>::set_parallel_loop4(
	const std::vector<TA> &atoms_vec)
{
	constexpr std::size_t num_index = 4;
	const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,TC>>>>
		atoms_split_list = Distribute_Equally_Weighted::distribute_atoms_periods(
			this->mpi_comm, atoms_vec, this->period, num_index, false, this->atoms_weight);

	this->list_Aa01 = atoms_split_list.first;
	this->list_Aa2  = atoms_split_list.second[0];
	this->list_Ab01 = atoms_split_list.second[1];
	this->list_Ab2  = atoms_split_list.second[2];
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void Parallel_LRI_Equally_Weighted<TA,Tcell,Ndim,Tdata>::set_parallel_loop3(
	const std::vector<TA> &atoms_vec,
	const std::set<Label::Aab_Aab> &labels)
{
	constexpr std::size_t num_index = 2;
	const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,TC>>>>
		atoms_split_list1 = Distribute_Equally_Weighted::distribute_atoms_periods(
			this->mpi_comm, atoms_vec, this->period, num_index, false, this->atoms_weight);
	const std::vector<std::vector<std::pair<TA,TC>>>
		atoms_split_list2 = Distribute_Equally_Weighted::distribute_periods(
			this->mpi_comm, atoms_vec, this->period, num_index, false, this->atoms_weight);
	
	this->set_atoms_loop3(atoms_vec, atoms_split_list1, atoms_split_list2, labels);
}

}