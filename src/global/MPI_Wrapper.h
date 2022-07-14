// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include <mpi.h>

namespace MPI_Wrapper
{		
	int mpi_get_rank(const MPI_Comm &mpi_comm)
	{
		int rank_mine;
		MPI_Comm_rank (mpi_comm, &rank_mine);
		return rank_mine;
	}
		
	int mpi_get_size(const MPI_Comm &mpi_comm)
	{
		int rank_size;
		MPI_Comm_size (mpi_comm, &rank_size);
		return rank_size;
	}

	int mpi_get_count(MPI_Status* status, const MPI_Datatype &datatype)
	{
		int count;
		MPI_Get_count(status, datatype, &count);
		return count;
	}	
}