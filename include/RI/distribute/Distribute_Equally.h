// ===================
//  Author: Peize Lin
//  date: 2022.07.15
// ===================

#pragma once

#include "Distribute_Equally.hpp"

#include <vector>
#include <array>
#include <utility>
#include <mpi.h>

namespace RI
{

namespace Distribute_Equally
{
	// num_index 维张量，第0维为atoms，剩余维为{atom,cell}。
	// 返回值为每维上本进程被分配到的值列表，first为第0维，second[i-1]为第i维。

	// 全部维按照atoms，尽可能均分
	template<typename TA, typename Tcell, size_t Ndim>
	extern std::pair<std::vector<TA>,
	                 std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
	distribute_atoms(
		const MPI_Comm &mpi_comm,
		const std::vector<TA> &atoms,
		const std::array<Tcell,Ndim> &period,
		const size_t num_index);

	// 第0维按照atoms、剩余维按照{atom,period}，尽可能均分。
	template<typename TA, typename Tcell, size_t Ndim>
	extern std::pair<std::vector<TA>,
	                 std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
	distribute_atoms_periods(
		const MPI_Comm &mpi_comm,
		const std::vector<TA> &atoms,
		const std::array<Tcell,Ndim> &period,
		const size_t num_index);
}

}