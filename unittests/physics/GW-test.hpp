// ===================
//  Author: Minye Zhang, almost copied from Exx-test.hpp
//  date: 2022.06.02
// ===================

#pragma once

#include "RI/physics/GW.h"
// #include <complex>

namespace GW_Test
{
	template<typename Tdata>
	void main(int argc, char *argv[])
	{
		int mpi_init_provide;
		MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_init_provide);

		RI::G0W0<int,int,1,Tdata> g0w0;
		g0w0.set_parallel(MPI_COMM_WORLD, {{1,{0}},{2,{4}}}, {}, {1});
		g0w0.set_Cs({}, 1E-4);
		g0w0.set_Ws({}, 1E-4);
		g0w0.set_Gs({}, 1E-4);
		g0w0.cal_Sigmas();

		g0w0.free_Cs();
		g0w0.free_Ws();
		g0w0.free_Gs();

		MPI_Finalize();
	}
}