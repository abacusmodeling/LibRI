// ===================
//  Author: Peize Lin
//  date: 2022.05.01
// ===================

#pragma once

#include "global/Cereal_Func.h"
#include "unittests/print_stl.h"

#include <vector>
#include <map>
#include <string>

#include <unistd.h>

namespace Cereal_Test
{
	static void main()
	{
		MPI_Init(NULL,NULL);
		int rank_size;	MPI_Comm_size( MPI_COMM_WORLD, &rank_size );
		int rank_mine;	MPI_Comm_rank( MPI_COMM_WORLD, &rank_mine );

		if(rank_mine==0)
		{
			std::vector<double> v = {1,2,3,4,5};
			std::map<int,double> m = {{1,2.3}, {4,5.6}, {-7,-8.9}};
			std::stringstream ss;
			MPI_Request request;
			Cereal_Func::mpi_isend(1, 0, MPI_COMM_WORLD, ss, request,
				v, std::string("abc"), -100, m);

			std::cout<<"#\t"<<ss.str().size()<<std::endl;
			int flag=0;
			do
			{
				MPI_Test( &request, &flag, MPI_STATUS_IGNORE );
				std::cout<<"f"<<flag<<std::flush;
			} while (!flag);
		}
		else if(rank_mine==1)
		{
			std::vector<double> v;
			std::string s;
			int i;
			std::map<int,double> m;
			MPI_Status status = Cereal_Func::mpi_recv(MPI_COMM_WORLD,
				v, s, i, m);

			std::cout<<"@\t"<<v<<std::endl;
			std::cout<<"@\t"<<s<<std::endl;
			std::cout<<"@\t"<<i<<std::endl;
			std::cout<<"@\t"<<m<<std::endl;

			int status_size;	MPI_Get_count( &status, MPI_CHAR, &status_size);
			std::cout<<"@\t"<<status.MPI_SOURCE<<"\t"<<status.MPI_TAG<<"\t"<<status.MPI_ERROR<<"\t"<<status_size<<std::endl;
		}
		MPI_Finalize();
	}
}