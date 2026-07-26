// ===================
//  Author: maki49
//  date: 2026.07.10
// ===================

#pragma once

#include "Distribute_Equally_Weighted.h"
#include "Split_Processes.h"
#include "Divide_Atoms_Weighted.h"

namespace RI
{

namespace Distribute_Equally_Weighted
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
		const bool flag_task_repeatable,
		const std::map<TA,std::size_t> &atoms_weight)
	{
		assert(num_index>=1);
		using TAC = std::pair<TA,std::array<Tcell,Ndim>>;

		// task_sizes 保持按个数统计：每一维用的是同一个原子集合，改按 sum(weight) 统计
		// 只会给所有维乘上同一个公共因子，split_first 里 group_size 取的是它们的比值，
		// 进程网格形状不变。只有下面 Divide_Atoms_Weighted 的组内划分才需要加权。
		const std::vector<std::size_t> task_sizes(num_index, atoms.size());
		const std::vector<std::tuple<MPI_Wrapper::mpi_comm, std::size_t, std::size_t>>
			comm_color_sizes = Split_Processes::split_all(mpi_comm, task_sizes);

		std::pair<std::vector<TA>, std::vector<std::vector<TAC>>> atoms_split_list;
		atoms_split_list.second.resize(num_index-1);

		if(!flag_task_repeatable)
			if(RI::MPI_Wrapper::mpi_get_rank(std::get<0>(comm_color_sizes.back())()))
				return atoms_split_list;

		atoms_split_list.first = Divide_Atoms_Weighted::divide_atoms(
			std::get<1>(comm_color_sizes[1]),
			std::get<2>(comm_color_sizes[1]),
			atoms,
			atoms_weight);
		for(std::size_t i=1; i<num_index; ++i)
			atoms_split_list.second[i-1] = Divide_Atoms_Weighted::divide_atoms(
				std::get<1>(comm_color_sizes[i+1]),
				std::get<2>(comm_color_sizes[i+1]),
				atoms,
				period,
				atoms_weight);


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
		const bool flag_task_repeatable,
		const std::map<TA,std::size_t> &atoms_weight)
	{
		assert(num_index>=1);
		using TAC = std::pair<TA,std::array<Tcell,Ndim>>;

		// task_sizes 保持按个数统计，理由见 distribute_atoms
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

		atoms_split_list.first = Divide_Atoms_Weighted::divide_atoms(
			std::get<1>(comm_color_sizes[1]),
			std::get<2>(comm_color_sizes[1]),
			atoms,
			atoms_weight);
		for(std::size_t i=1; i<num_index; ++i)
			atoms_split_list.second[i-1] = Divide_Atoms_Weighted::divide_atoms_periods(
				std::get<1>(comm_color_sizes[i+1]),
				std::get<2>(comm_color_sizes[i+1]),
				atoms,
				period,
				atoms_weight);
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
		const bool flag_task_repeatable,
		const std::map<TA,std::size_t> &atoms_weight)
	{
		assert(num_index>=1);
		using TAC = std::pair<TA,std::array<Tcell,Ndim>>;

		// task_sizes 保持按个数统计，理由见 distribute_atoms
		const std::size_t task_size_period = atoms.size() * std::accumulate( period.begin(), period.end(), 1, std::multiplies<Tcell>() );
		std::vector<std::size_t> task_sizes(num_index, task_size_period);
		const std::vector<std::tuple<MPI_Wrapper::mpi_comm, std::size_t, std::size_t>>
			comm_color_sizes = Split_Processes::split_all(mpi_comm, task_sizes);

		std::vector<std::vector<TAC>> atoms_split_list(num_index);

		if(!flag_task_repeatable)
			if(RI::MPI_Wrapper::mpi_get_rank(std::get<0>(comm_color_sizes.back())()))
				return atoms_split_list;

		for(std::size_t i=0; i<num_index; ++i)
			atoms_split_list[i] = Divide_Atoms_Weighted::divide_atoms_periods(
				std::get<1>(comm_color_sizes[i+1]),
				std::get<2>(comm_color_sizes[i+1]),
				atoms,
				period,
				atoms_weight);
		return atoms_split_list;
	}
}

}