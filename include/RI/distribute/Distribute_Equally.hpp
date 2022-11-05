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
	template<typename TA, typename Tcell, size_t Ndim>
	std::pair<std::vector<TA>,
	          std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
	distribute_atoms(
		const MPI_Comm &mpi_comm,
		const std::vector<TA> &atoms,
		const std::array<Tcell,Ndim> &period,
		const size_t num_index)
	{
		assert(num_index>=1);
		using TAC = std::pair<TA,std::array<Tcell,Ndim>>;

		const std::vector<int> Ns(num_index, atoms.size());
		const std::vector<std::tuple<MPI_Comm,int,int>> comm_color_sizes = Split_Processes::split_all(mpi_comm, Ns);

		std::pair<std::vector<TA>, std::vector<std::vector<TAC>>> atoms_split_list;
		atoms_split_list.first = Divide_Atoms::divide_atoms(
			std::get<1>(comm_color_sizes[1]),
			std::get<2>(comm_color_sizes[1]),
			atoms);
		atoms_split_list.second.resize(num_index-1);
		for(int i=1; i<num_index; ++i)
			atoms_split_list.second[i-1] = Divide_Atoms::divide_atoms(
				std::get<1>(comm_color_sizes[i+1]),
				std::get<2>(comm_color_sizes[i+1]),
				atoms,
				period);
		return atoms_split_list;
	}

	template<typename TA, typename Tcell, size_t Ndim>
	std::pair<std::vector<TA>,
	          std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
	distribute_atoms_periods(
		const MPI_Comm &mpi_comm,
		const std::vector<TA> &atoms,
		const std::array<Tcell,Ndim> &period,
		const size_t num_index)
	{
		assert(num_index>=1);
		using TAC = std::pair<TA,std::array<Tcell,Ndim>>;

		std::vector<int> Ns(num_index);
		Ns[0] = atoms.size();
		for(size_t i=1; i<num_index; ++i)
			Ns[i] =
				atoms.size()
				* std::accumulate( period.begin(), period.end(), 1, std::multiplies<Tcell>() );
		const std::vector<std::tuple<MPI_Comm,int,int>> comm_color_sizes = Split_Processes::split_all(mpi_comm, Ns);

		std::pair<std::vector<TA>, std::vector<std::vector<TAC>>> atoms_split_list;
		atoms_split_list.first = Divide_Atoms::divide_atoms(
			std::get<1>(comm_color_sizes[1]),
			std::get<2>(comm_color_sizes[1]),
			atoms);
		atoms_split_list.second.resize(num_index-1);
		for(int i=1; i<num_index; ++i)
			atoms_split_list.second[i-1] = Divide_Atoms::divide_atoms_periods(
				std::get<1>(comm_color_sizes[i+1]),
				std::get<2>(comm_color_sizes[i+1]),
				atoms,
				period);
		return atoms_split_list;
	}
}

}