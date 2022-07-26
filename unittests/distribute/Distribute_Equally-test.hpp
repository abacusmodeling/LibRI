// ===================
//  Author: Peize Lin
//  date: 2022.07.15
// ===================

#pragma once

#include "RI/distribute/Distribute_Equally.h"
#include "RI/global/MPI_Wrapper.h"
#include "unittests/print_stl.h"

#include <fstream>

namespace Distribute_Equally_Test
{
	void test_distribute_atoms()
	{
		MPI_Init(NULL,NULL);

		const std::vector<int> atoms = {0,1,2,3,4};
		const std::array<int,1> period = {2};
		const size_t num_index = 2;
		const std::pair<std::vector<int>,
		                std::vector<std::vector<std::pair<int,std::array<int,1>>>>>
			atoms_split_list
			= Distribute_Equally::distribute_atoms(MPI_COMM_WORLD, atoms, period, num_index);
		
		std::ofstream ofs("out."+std::to_string(MPI_Wrapper::mpi_get_rank(MPI_COMM_WORLD)));
		ofs<<atoms_split_list.first<<std::endl;
		for(const auto &a : atoms_split_list.second)
			ofs<<a<<std::endl;

		MPI_Finalize();	
	}
	/*
	mpirun -n 7
		rank 0
			0|	1|	
			{ 0, 0	 }|	{ 0, 1	 }|	{ 1, 0	 }|	{ 1, 1	 }|	
		rank 1
			0|	1|	
			{ 2, 0	 }|	{ 2, 1	 }|	{ 3, 0	 }|	{ 3, 1	 }|	
		rank 2
			0|	1|	
			{ 4, 0	 }|	{ 4, 1	 }|	
		rank 3
			2|	3|	
			{ 0, 0	 }|	{ 0, 1	 }|	{ 1, 0	 }|	{ 1, 1	 }|	{ 2, 0	 }|	{ 2, 1	 }|	
		rank 4
			2|	3|	
			{ 3, 0	 }|	{ 3, 1	 }|	{ 4, 0	 }|	{ 4, 1	 }|	
		rank 5
			4|	
			{ 0, 0	 }|	{ 0, 1	 }|	{ 1, 0	 }|	{ 1, 1	 }|	{ 2, 0	 }|	{ 2, 1	 }|	
		rank 6
			4|	
			{ 3, 0	 }|	{ 3, 1	 }|	{ 4, 0	 }|	{ 4, 1	 }|	
	*/

	void test_distribute_atoms_periods()
	{
		MPI_Init(NULL,NULL);

		const std::vector<int> atoms = {0,1,2,3,4};
		const std::array<int,1> period = {2};
		const size_t num_index = 2;
		const std::pair<std::vector<int>, 
		                std::vector<std::vector<std::pair<int,std::array<int,1>>>>>
			atoms_split_list
			= Distribute_Equally::distribute_atoms_periods(MPI_COMM_WORLD, atoms, period, num_index);
		
		std::ofstream ofs("out."+std::to_string(MPI_Wrapper::mpi_get_rank(MPI_COMM_WORLD)));
		ofs<<atoms_split_list.first<<std::endl;
		for(const auto &a : atoms_split_list.second)
			ofs<<a<<std::endl;

		MPI_Finalize();	
	}
	/*
	mpirun -n 7
		rank 0
			0|	1|	
			{ 0, 0	 }|	{ 0, 1	 }|	{ 1, 0	 }|	{ 1, 1	 }|	
		rank 1
			0|	1|	
			{ 2, 0	 }|	{ 2, 1	 }|	{ 3, 0	 }|	
		rank 2
			0|	1|	
			{ 3, 1	 }|	{ 4, 0	 }|	{ 4, 1	 }|	
		rank 3
			2|	3|	
			{ 0, 0	 }|	{ 0, 1	 }|	{ 1, 0	 }|	{ 1, 1	 }|	{ 2, 0	 }|	
		rank 4
			2|	3|	
			{ 2, 1	 }|	{ 3, 0	 }|	{ 3, 1	 }|	{ 4, 0	 }|	{ 4, 1	 }|	
		rank 5
			4|	
			{ 0, 0	 }|	{ 0, 1	 }|	{ 1, 0	 }|	{ 1, 1	 }|	{ 2, 0	 }|	
		rank 6
			4|	
			{ 2, 1	 }|	{ 3, 0	 }|	{ 3, 1	 }|	{ 4, 0	 }|	{ 4, 1	 }|	
	*/	
}