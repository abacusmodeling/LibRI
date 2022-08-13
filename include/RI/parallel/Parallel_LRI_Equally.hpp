// ===================
//  Author: Peize Lin
//  date: 2022.07.23
// ===================

#pragma once

#include "Parallel_LRI_Equally.h"
#include "../global/Global_Func-1.h"
#include "../distribute/Distribute_Equally.h"
#include "../comm/mix/Communicate_Tensors_Map_Judge.h"

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Parallel_LRI_Equally<TA,Tcell,Ndim,Tdata>::set_parallel(
	const MPI_Comm &mpi_comm_in,
	const std::map<TA,TatomR> &atomsR,
	const std::array<TatomR,Ndim> &latvec,
	const std::array<Tcell,Ndim> &period_in)
{
	this->mpi_comm = mpi_comm_in;
	this->period = period_in;

	constexpr size_t num_index = 4;
	const std::vector<TA> atoms_vec = Global_Func::map_key_to_vec(atomsR);

	std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,TC>>>>
		atoms_split_list = Distribute_Equally::distribute_atoms_periods(
			mpi_comm, atoms_vec, period, num_index);
			
	this->list_Aa01 = atoms_split_list.first;
	this->list_Aa2  = atoms_split_list.second[0];
	this->list_Ab01 = atoms_split_list.second[1];
	this->list_Ab2  = atoms_split_list.second[2];
}


template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
auto Parallel_LRI_Equally<TA,Tcell,Ndim,Tdata>::comm_tensors_map2(
	const Label::ab &label,
	const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds) const
-> std::map<TA,std::map<TAC,Tensor<Tdata>>>
{
	switch(label)
	{
		case Label::ab::a:
			return Communicate_Tensors_Map_Judge::comm_map2(this->mpi_comm, Ds, Global_Func::to_set(this->list_Aa01), Global_Func::to_set(this->list_Aa2));
		case Label::ab::b:
			return Communicate_Tensors_Map_Judge::comm_map2_period(this->mpi_comm, Ds, Global_Func::to_set(this->list_Ab01), Global_Func::to_set(this->list_Ab2), this->period);
		case Label::ab::a0b0:	case Label::ab::a0b1:
		case Label::ab::a1b0:	case Label::ab::a1b1:
			return Communicate_Tensors_Map_Judge::comm_map2(this->mpi_comm, Ds, Global_Func::to_set(this->list_Aa01), Global_Func::to_set(this->list_Ab01));
		case Label::ab::a0b2:
		case Label::ab::a1b2:
			return Communicate_Tensors_Map_Judge::comm_map2(this->mpi_comm, Ds, Global_Func::to_set(this->list_Aa01), Global_Func::to_set(this->list_Ab2));
		case Label::ab::a2b0:	case Label::ab::a2b1:
			return Communicate_Tensors_Map_Judge::comm_map2_period(this->mpi_comm, Ds, Global_Func::to_set(this->list_Aa2), Global_Func::to_set(this->list_Ab01), this->period);
		case Label::ab::a2b2:
			return Communicate_Tensors_Map_Judge::comm_map2_period(this->mpi_comm, Ds, Global_Func::to_set(this->list_Aa2), Global_Func::to_set(this->list_Ab2), this->period);
		default:
			throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
	}
}