// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "RI/physics/Exx.h"
#include <complex>

namespace Exx_Test
{
	template<typename Tdata>
	void main(int argc, char *argv[])
	{
		int mpi_init_provide;
		MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_init_provide);

		Exx<int,int,1,Tdata> exx;
		exx.set_parallel(MPI_COMM_WORLD, {{1,{0}},{2,{4}}}, {}, {1});
		exx.set_Cs({}, 0);
		exx.set_Vs({}, 0);
		exx.set_Ds({}, 0);
		exx.cal_Hs();

		MPI_Finalize();
	}
}