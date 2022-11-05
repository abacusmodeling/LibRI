// ===================
//  Author: Peize Lin
//  date: 2022.07.11
// ===================

#pragma once

#include "RI/distribute/Split_Processes.h"

#include <fstream>

namespace Split_Processes_Test
{
	void test_split_all(int argc, char *argv[], const std::vector<int> &Ns)
	{
		MPI_Init(&argc, &argv);

		const std::vector<std::tuple<MPI_Comm,int,int>> comm_color_sizes = RI::Split_Processes::split_all(MPI_COMM_WORLD, Ns);

		std::ofstream ofs("out."+std::to_string(RI::MPI_Wrapper::mpi_get_rank(MPI_COMM_WORLD)), std::ofstream::app);
		for(const auto &comm_color_size : comm_color_sizes)
			ofs<<RI::MPI_Wrapper::mpi_get_rank(std::get<0>(comm_color_size))<<"\t"
			<<RI::MPI_Wrapper::mpi_get_size(std::get<0>(comm_color_size))<<"\t|\t"
			<<std::get<1>(comm_color_size)<<"\t"
			<<std::get<2>(comm_color_size)<<std::endl;

		MPI_Finalize();
	}

	void test_split_all(int argc, char *argv[])
	{
		//test_split_all(argc, argv, {4,5,6});
		/*
		mpirun -n 7
			rank 0
				0	7	|	0	1
				0	4	|	0	2
				0	2	|	0	2
				0	1	|	0	2
			rank 1
				1	7	|	0	1
				1	4	|	0	2
				1	2	|	0	2
				0	1	|	1	2
			rank 2
				2	7	|	0	1
				2	4	|	0	2
				0	2	|	1	2
				0	1	|	0	2
			rank 3
				3	7	|	0	1
				3	4	|	0	2
				1	2	|	1	2
				0	1	|	1	2
			rank 4
				4	7	|	0	1
				0	3	|	1	2
				0	2	|	0	2
				0	1	|	0	2
			rank 5
				5	7	|	0	1
				1	3	|	1	2
				1	2	|	0	2
				0	1	|	1	2
			rank 6
				6	7	|	0	1
				2	3	|	1	2
				0	1	|	1	2
				0	1	|	0	1
		*/

		//test_split_all(argc, argv, {4,5,30});

		test_split_all(argc, argv, {2});
		/*
		mpirun -n 7
		*/
	}
}