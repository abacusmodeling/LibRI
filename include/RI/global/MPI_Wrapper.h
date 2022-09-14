// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include <mpi.h>
#include <complex>

#define MPI_CHECK(x) if((x)!=MPI_SUCCESS)	throw std::runtime_error(std::string(__FILE__)+" line "+std::to_string(__LINE__));

namespace MPI_Wrapper
{		
	inline int mpi_get_rank(const MPI_Comm &mpi_comm)
	{
		int rank_mine;
		MPI_CHECK( MPI_Comm_rank (mpi_comm, &rank_mine) );
		return rank_mine;
	}
		
	inline int mpi_get_size(const MPI_Comm &mpi_comm)
	{
		int rank_size;
		MPI_CHECK( MPI_Comm_size (mpi_comm, &rank_size) );
		return rank_size;
	}

	inline int mpi_get_count(MPI_Status* status, const MPI_Datatype &datatype)
	{
		int count;
		MPI_CHECK( MPI_Get_count(status, datatype, &count) );
		return count;
	}

	inline void mpi_reduce(const float*const sendbuf, float*const recvbuf, const int count,
		const MPI_Op op, const int root, const MPI_Comm &mpi_comm)
	{
		MPI_CHECK( MPI_Reduce( sendbuf, recvbuf, count, MPI_FLOAT, op, root, mpi_comm ) );
	}
	inline void mpi_reduce(const double*const sendbuf, double*const recvbuf, const int count,
		const MPI_Op op, const int root, const MPI_Comm &mpi_comm)
	{
		MPI_CHECK( MPI_Reduce( sendbuf, recvbuf, count, MPI_DOUBLE, op, root, mpi_comm ) );
	}
	inline void mpi_reduce(const std::complex<float>*const sendbuf, std::complex<float>*const recvbuf, const int count,
		const MPI_Op op, const int root, const MPI_Comm &mpi_comm)
	{
		MPI_CHECK( MPI_Reduce( sendbuf, recvbuf, count, MPI_COMPLEX, op, root, mpi_comm ) );
	}
	inline void mpi_reduce(const std::complex<double>*const sendbuf, std::complex<double>*const recvbuf, const int count,
		const MPI_Op op, const int root, const MPI_Comm &mpi_comm)
	{
		MPI_CHECK( MPI_Reduce( sendbuf, recvbuf, count, MPI_DOUBLE_COMPLEX, op, root, mpi_comm ) );
	}
	template<typename T>
	inline void mpi_reduce(T &data, const MPI_Op op, const int root, const MPI_Comm &mpi_comm)
	{
		T data_tmp;
		mpi_reduce(&data, &data_tmp, 1, op, root, mpi_comm);
		data = data_tmp;
	}

	inline void mpi_allreduce(const float*const sendbuf, float*const recvbuf, const int count,
		const MPI_Op op, const MPI_Comm &mpi_comm)
	{
		MPI_CHECK( MPI_Allreduce( sendbuf, recvbuf, count, MPI_FLOAT, op, mpi_comm ) );
	}
	inline void mpi_allreduce(const double*const sendbuf, double*const recvbuf, const int count,
		const MPI_Op op, const MPI_Comm &mpi_comm)
	{
		MPI_CHECK( MPI_Allreduce( sendbuf, recvbuf, count, MPI_DOUBLE, op, mpi_comm ) );
	}
	inline void mpi_allreduce(const std::complex<float>*const sendbuf, std::complex<float>*const recvbuf, const int count,
		const MPI_Op op, const MPI_Comm &mpi_comm)
	{
		MPI_CHECK( MPI_Allreduce( sendbuf, recvbuf, count, MPI_COMPLEX, op, mpi_comm ) );
	}
	inline void mpi_allreduce(const std::complex<double>*const sendbuf, std::complex<double>*const recvbuf, const int count,
		const MPI_Op op, const MPI_Comm &mpi_comm)
	{
		MPI_CHECK( MPI_Allreduce( sendbuf, recvbuf, count, MPI_DOUBLE_COMPLEX, op, mpi_comm ) );
	}
	template<typename T>
	inline void mpi_allreduce(T &data, const MPI_Op op, const MPI_Comm &mpi_comm)
	{
		T data_tmp;
		mpi_allreduce(&data, &data_tmp, 1, op, mpi_comm);
		data = data_tmp;
	}
}

#undef MPI_CHECK