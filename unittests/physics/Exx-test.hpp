// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "physics/Exx.h"
#include <complex>

namespace Exx_Test
{
	template<typename Tdata>
	void main()
	{
		int mpi_init_provide;
		MPI_Init_thread(NULL,NULL, MPI_THREAD_MULTIPLE, &mpi_init_provide);

		Exx<int,int,1,std::complex<double>> exx(MPI_COMM_WORLD);
		exx.set_stru({{1,{0}},{2,{4}}}, {}, {1});
		exx.set_Cs({}, 0);
		exx.set_Vs({}, 0);
		exx.set_Ds({}, 0);
		exx.cal_Hs();

		MPI_Finalize();
	}
}