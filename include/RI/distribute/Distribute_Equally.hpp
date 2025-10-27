// ===================
//  Author: Peize Lin
//  date: 2022.07.15
// ===================

#pragma once

#include "Distribute_Equally.h"
#include "Split_Processes.h"
#include "Divide_Atoms.h"

namespace RI
{

namespace Distribute_Equally
{
	// 全部维按照atoms，尽可能均分
	template<typename TA, typename Tcell, std::size_t Ndim>
	std::pair<std::vector<TA>,
	          std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
	distribute_atoms(
		const MPI_Comm &mpi_comm,
		const std::vector<TA> &atoms,
		const std::array<Tcell,Ndim> &period,
		const std::size_t num_index,
		const bool flag_task_repeatable)
	{
		assert(num_index>=1);
		using TAC = std::pair<TA,std::array<Tcell,Ndim>>;

		const std::vector<std::size_t> task_sizes(num_index, atoms.size());
		const std::vector<std::tuple<MPI_Wrapper::mpi_comm, std::size_t, std::size_t>>
			comm_color_sizes = Split_Processes::split_all(mpi_comm, task_sizes);

		std::pair<std::vector<TA>, std::vector<std::vector<TAC>>> atoms_split_list;
		atoms_split_list.second.resize(num_index-1);

		if(!flag_task_repeatable)
			if(RI::MPI_Wrapper::mpi_get_rank(std::get<0>(comm_color_sizes.back())()))
				return atoms_split_list;

		atoms_split_list.first = Divide_Atoms::divide_atoms(
			std::get<1>(comm_color_sizes[1]),
			std::get<2>(comm_color_sizes[1]),
			atoms);
		for(std::size_t i=1; i<num_index; ++i)
			atoms_split_list.second[i-1] = Divide_Atoms::divide_atoms(
				std::get<1>(comm_color_sizes[i+1]),
				std::get<2>(comm_color_sizes[i+1]),
				atoms,
				period);


		return atoms_split_list;
	}

	// 第0维按照atoms、剩余维按照{atom,period}，尽可能均分
	template<typename TA, typename Tcell, std::size_t Ndim>
	std::pair<std::vector<TA>,
	          std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
	distribute_atoms_periods(
		const MPI_Comm &mpi_comm,
		const std::vector<TA> &atoms,
		const std::array<Tcell,Ndim> &period,
		const std::size_t num_index,
		const bool flag_task_repeatable)
	{
		assert(num_index>=1);
		using TAC = std::pair<TA,std::array<Tcell,Ndim>>;

		const std::size_t task_size_period = atoms.size() * std::accumulate( period.begin(), period.end(), 1, std::multiplies<Tcell>() );
		std::vector<std::size_t> task_sizes(num_index, task_size_period);
		task_sizes[0] = atoms.size();
		const std::vector<std::tuple<MPI_Wrapper::mpi_comm, std::size_t, std::size_t>>
			comm_color_sizes = Split_Processes::split_all(mpi_comm, task_sizes);

		std::pair<std::vector<TA>, std::vector<std::vector<TAC>>> atoms_split_list;
		atoms_split_list.second.resize(num_index-1);

		if(!flag_task_repeatable)
			if(RI::MPI_Wrapper::mpi_get_rank(std::get<0>(comm_color_sizes.back())()))
				return atoms_split_list;

		atoms_split_list.first = Divide_Atoms::divide_atoms(
			std::get<1>(comm_color_sizes[1]),
			std::get<2>(comm_color_sizes[1]),
			atoms);
		for(std::size_t i=1; i<num_index; ++i)
			atoms_split_list.second[i-1] = Divide_Atoms::divide_atoms_periods(
				std::get<1>(comm_color_sizes[i+1]),
				std::get<2>(comm_color_sizes[i+1]),
				atoms,
				period);
		return atoms_split_list;
	}

	// 全部维按照{atom,period}，尽可能均分
	template<typename TA, typename Tcell, std::size_t Ndim>
	extern std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>
	distribute_periods(
		const MPI_Comm &mpi_comm,
		const std::vector<TA> &atoms,
		const std::array<Tcell,Ndim> &period,
		const std::size_t num_index,
		const bool flag_task_repeatable)
	{
		assert(num_index>=1);
		using TAC = std::pair<TA,std::array<Tcell,Ndim>>;

		const std::size_t task_size_period = atoms.size() * std::accumulate( period.begin(), period.end(), 1, std::multiplies<Tcell>() );
		std::vector<std::size_t> task_sizes(num_index, task_size_period);
		const std::vector<std::tuple<MPI_Wrapper::mpi_comm, std::size_t, std::size_t>>
			comm_color_sizes = Split_Processes::split_all(mpi_comm, task_sizes);

		std::vector<std::vector<TAC>> atoms_split_list(num_index);

		if(!flag_task_repeatable)
			if(RI::MPI_Wrapper::mpi_get_rank(std::get<0>(comm_color_sizes.back())()))
				return atoms_split_list;

		for(std::size_t i=0; i<num_index; ++i)
			atoms_split_list[i] = Divide_Atoms::divide_atoms_periods(
				std::get<1>(comm_color_sizes[i+1]),
				std::get<2>(comm_color_sizes[i+1]),
				atoms,
				period);
		return atoms_split_list;
	}

	// 均分{atomI,atomJ,k1,k2}
	template<typename Tindex>
	void distribute_atom_and_k_pair(
		const MPI_Comm &mpi_comm,
		const std::size_t nat,
		const std::size_t nk,
		std::vector<Tindex> &list_I,
		std::vector<Tindex> &list_J,
		std::vector<Tindex> &list_k1_index,
		std::vector<Tindex> &list_k2_index,
		const bool flag_task_repeatable)
	{
		// task_sizes的顺序必须从小到大，否则在split中会出现rank_size<group_size，所以先判断nat和nk的大小
		std::size_t ntaskA, ntaskB;
		std::vector<Tindex> *A1_ptr, *A2_ptr, *B1_ptr, *B2_ptr;
		if (nk >= nat)
		{
			ntaskA = nat;
			ntaskB = nk;
			A1_ptr = &list_I;
			A2_ptr = &list_J;
			B1_ptr = &list_k1_index;
			B2_ptr = &list_k2_index;
		}
		else
		{
			ntaskA = nk;
			ntaskB = nat;
			A1_ptr = &list_k1_index;
			A2_ptr = &list_k2_index;
			B1_ptr = &list_I;
			B2_ptr = &list_J;
		}
		const std::vector<std::size_t> task_sizes{ntaskA, ntaskA, ntaskB, ntaskB};
		const std::vector<std::tuple<MPI_Wrapper::mpi_comm, std::size_t, std::size_t>>
			comm_color_sizes = Split_Processes::split_all(mpi_comm, task_sizes);

		if(!flag_task_repeatable)
			if(RI::MPI_Wrapper::mpi_get_rank(std::get<0>(comm_color_sizes.back())()))
				return;
		
		std::vector<Tindex> indicesA, indicesB;
		for(Tindex i=0; i<ntaskA; ++i)
			indicesA.push_back(i);
		for(Tindex i=0; i<ntaskB; ++i)
			indicesB.push_back(i);

		*A1_ptr = Divide_Atoms::divide_atoms(
			std::get<1>(comm_color_sizes[1]),
			std::get<2>(comm_color_sizes[1]),
			indicesA);
		*A2_ptr = Divide_Atoms::divide_atoms(
			std::get<1>(comm_color_sizes[2]),
			std::get<2>(comm_color_sizes[2]),
			indicesA);
		*B1_ptr = Divide_Atoms::divide_atoms(
			std::get<1>(comm_color_sizes[3]),
			std::get<2>(comm_color_sizes[3]),
			indicesB);
		*B2_ptr = Divide_Atoms::divide_atoms(
			std::get<1>(comm_color_sizes[4]),
			std::get<2>(comm_color_sizes[4]),
			indicesB);
	}

	// 均分{atomI,atomJ,k}
	template<typename Tindex>
	void distribute_atom_pair_and_k(
		const MPI_Comm &mpi_comm,
		const std::size_t nat,
		const std::size_t nk,
		std::vector<Tindex> &list_I,
		std::vector<Tindex> &list_J,
		std::vector<Tindex> &list_k_index,
		const bool flag_task_repeatable)
	{
		std::vector<std::size_t> task_sizes;
		std::vector<Tindex> indices_atom, indices_k;
		for(Tindex i=0; i<nat; ++i)
			indices_atom.push_back(i);
		for(Tindex i=0; i<nk; ++i)
			indices_k.push_back(i);
	// task_sizes的顺序必须从小到大，否则在split中会出现rank_size<group_size，所以先判断nat和nk的大小
		if (nk >= nat)
		{
			task_sizes = {nat, nat, nk};
		}
		else
		{
			task_sizes = {nk, nat, nat};
		}
		const std::vector<std::tuple<MPI_Wrapper::mpi_comm, std::size_t, std::size_t>>
			comm_color_sizes = Split_Processes::split_all(mpi_comm, task_sizes);

		if(!flag_task_repeatable)
			if(RI::MPI_Wrapper::mpi_get_rank(std::get<0>(comm_color_sizes.back())()))
				return;
		
		if (nk >= nat)
		{
			list_I = Divide_Atoms::divide_atoms(
				std::get<1>(comm_color_sizes[1]),
				std::get<2>(comm_color_sizes[1]),
				indices_atom);
			list_J = Divide_Atoms::divide_atoms(
				std::get<1>(comm_color_sizes[2]),
				std::get<2>(comm_color_sizes[2]),
				indices_atom);
			list_k_index = Divide_Atoms::divide_atoms(
				std::get<1>(comm_color_sizes[3]),
				std::get<2>(comm_color_sizes[3]),
				indices_k);
		}
		else
		{
			list_k_index = Divide_Atoms::divide_atoms(
				std::get<1>(comm_color_sizes[1]),
				std::get<2>(comm_color_sizes[1]),
				indices_k);
			list_I = Divide_Atoms::divide_atoms(
				std::get<1>(comm_color_sizes[2]),
				std::get<2>(comm_color_sizes[2]),
				indices_atom);
			list_J = Divide_Atoms::divide_atoms(
				std::get<1>(comm_color_sizes[3]),
				std::get<2>(comm_color_sizes[3]),
				indices_atom);
		}
	}
}

}