// ===================
//  Author: Peize Lin
//  date: 2022.07.11
// ===================

#pragma once

#include "Split_Processes.h"
#include "global/MPI_Wrapper.h"

#include <numeric>
#include <cassert>
#include <string>
#include <stdexcept>

#define MPI_CHECK(x) if((x)!=MPI_SUCCESS)	throw std::runtime_error(std::string(__FILE__)+" line "+std::to_string(__LINE__));

namespace Split_Processes
{
	// comm_color
	std::tuple<MPI_Comm,int> split(const MPI_Comm &mpi_comm, const int &group_size)
	{
		assert(group_size>0);
		const int rank_mine = MPI_Wrapper::mpi_get_rank(mpi_comm);
		const int rank_size = MPI_Wrapper::mpi_get_size(mpi_comm);
		assert(rank_size>=group_size);

		std::vector<int> num(group_size);			// sum(num) = rank_size
		const int mod = rank_size % group_size;
		for(int i=0; i<mod; ++i)
			num[i] = rank_size/group_size+1;
		for(int i=mod; i<group_size; ++i)
			num[i] = rank_size/group_size;

		std::vector<int> slice(group_size);			// slice.back() = rank_size
		slice[0] = num[0];
		for(int i=1; i<group_size; ++i)
			slice[i] = slice[i-1] + num[i];

		const int color_group = [&]() -> int		// which group should rank_mine in
		{
			for(int i=0; i<group_size; ++i)
				if(rank_mine < slice[i])
					return i;
			throw std::range_error(std::string(__FILE__)+" line "+std::to_string(__LINE__));
		}();

		MPI_Comm mpi_comm_split;
		MPI_CHECK( MPI_Comm_split( mpi_comm, color_group, rank_mine, &mpi_comm_split ) );

		return std::make_tuple(mpi_comm_split, color_group);
	}

	// comm_color_size
	std::tuple<MPI_Comm,int,int> split_first(const MPI_Comm &mpi_comm, const std::vector<int> &Ns)
	{
		assert(Ns.size()>=1);
		const int rank_size = MPI_Wrapper::mpi_get_size(mpi_comm);
		const int N_product = std::accumulate( Ns.begin(), Ns.end(), 1, std::multiplies<int>() );
		if(N_product>=rank_size)
		{
			const double num_average = std::pow(static_cast<double>(N_product)/rank_size, 1.0/Ns.size());
			const int group_size = std::round(Ns[0]/num_average);
			const std::tuple<MPI_Comm,int> comm_color = split(mpi_comm, group_size);
			return std::make_tuple(std::get<0>(comm_color), std::get<1>(comm_color), group_size);
		}
		else
		{
			if(Ns.size()>1)
			{
				const int group_size = Ns[0];
				const std::tuple<MPI_Comm,int> comm_color = split(mpi_comm, group_size);
				return std::make_tuple(std::get<0>(comm_color), std::get<1>(comm_color), group_size);
			}
			else
			{
				throw std::range_error(std::string(__FILE__)+" line "+std::to_string(__LINE__));
			}
		}
	}

	// vector<comm_color_size>
	std::vector<std::tuple<MPI_Comm,int,int>> split_all(const MPI_Comm &mpi_comm, const std::vector<int> &Ns)
	{
		std::vector<std::tuple<MPI_Comm,int,int>> comm_color_sizes(Ns.size()+1);
		comm_color_sizes[0] = std::make_tuple(mpi_comm, 0, 1);
		for(int m=0; m<Ns.size(); ++m)
			comm_color_sizes[m+1] = split_first(std::get<0>(comm_color_sizes[m]), {Ns.begin()+m, Ns.end()});
		return comm_color_sizes;
	}
}

#undef MPI_CHECK