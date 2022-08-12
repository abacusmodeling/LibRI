// ===================
//  Author: Peize Lin
//  date: 2022.07.11
// ===================

#pragma once

#include <mpi.h>
#include <tuple>
#include <vector>

namespace Split_Processes
{
	// comm_color
	extern std::tuple<MPI_Comm,int> split(const MPI_Comm &mpi_comm, const int &group_size);

	// comm_color_size
	extern std::tuple<MPI_Comm,int,int> split_first(const MPI_Comm &mpi_comm, const std::vector<int> &Ns);

	// vector<comm_color_size>
	extern std::vector<std::tuple<MPI_Comm,int,int>> split_all(const MPI_Comm &mpi_comm, const std::vector<int> &Ns);
}

#include "Split_Processes.hpp"