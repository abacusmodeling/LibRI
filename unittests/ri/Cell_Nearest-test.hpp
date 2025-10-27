// ===================
//  Author: Peize Lin
//  date: 2022.12.30
// ===================

#pragma once

#include "../print_stl.h"

#include "RI/ri/Cell_Nearest.h"
#include "../global/Tensor-test.h"

namespace Cell_Nearest_Test
{
	static void main()
	{
		using TA = std::string;
		using Tcell = int;
		constexpr int Ndim = 2;
		using Tpos = double;
		constexpr int Npos = 2;
		using Tatom_pos = std::array<Tpos,Npos>;

		std::map<TA,Tatom_pos> atoms_pos;
		std::array<Tatom_pos,Ndim> latvec;
		std::array<Tcell,Ndim> period;

		auto test = [&atoms_pos, &latvec, &period]()
		{
			RI::Cell_Nearest<TA, Tcell, Ndim, Tpos, Npos> stress;
			stress.init(atoms_pos, latvec, period);
			std::cout<<stress.cells_nearest_continuous<<std::endl;

			for(Tcell idim0=-1; idim0<period[0]+1; ++idim0)
				for(Tcell idim1=-1; idim1<period[1]+1; ++idim1)
					std::cout<<idim0<<"\t"<<idim1<<"\t|\t"<<stress.get_cell_nearest_discrete("H", "C", {idim0,idim1})<<std::endl;
			std::cout<<std::endl;
		};

		{
			atoms_pos["H"] = {0, 0};
			atoms_pos["C"] = {-8, 2};
			latvec[0]={10,0};
			latvec[1]={0,10};
			period = {2,1};
			test();
			/*
				C	C	0		0
					H	-0.8	0.2
				H	C	0.8		-0.2
					H	0		0

				-1	-1	|	1	0
				-1	0	|	1	0
				-1	1	|	1	0
				0	-1	|	0	0
				0	0	|	0	0
				0	1	|	0	0
				1	-1	|	1	0
				1	0	|	1	0
				1	1	|	1	0
				2	-1	|	0	0
				2	0	|	0	0
				2	1	|	0	0
			*/
		}

		{
			atoms_pos["H"] = {0, 0};
			atoms_pos["C"] = {8, 2};
			latvec[0]={10,0};
			latvec[1]={0,10};
			period = {2,1};
			test();
			/*
				C	C	0		0
					H	0.8		0.2
				H	C	-0.8	-0.2
					H	0		0

				-1	-1	|	-1	0
				-1	0	|	-1	0
				-1	1	|	-1	0
				0	-1	|	0	0
				0	0	|	0	0
				0	1	|	0	0
				1	-1	|	-1	0
				1	0	|	-1	0
				1	1	|	-1	0
				2	-1	|	0	0
				2	0	|	0	0
				2	1	|	0	0
			*/
		}

		{
			atoms_pos["H"] = {0, 0};
			atoms_pos["C"] = {18, 2};
			latvec[0]={10,0};
			latvec[1]={0,10};
			period = {2,1};
			test();
			/*
				C	C	0		0
					H	1.8		0.2

				H	C	-1.8	-0.2
					H	0		0

				-1	-1	|	-1	0
				-1	0	|	-1	0
				-1	1	|	-1	0
				0	-1	|	-2	0
				0	0	|	-2	0
				0	1	|	-2	0
				1	-1	|	-1	0
				1	0	|	-1	0
				1	1	|	-1	0
				2	-1	|	-2	0
				2	0	|	-2	0
				2	1	|	-2	0
			*/
		}

		{
			atoms_pos["H"] = {0, 0};
			atoms_pos["C"] = {8.3, 3.4};
			latvec[0]={10,0};
			latvec[1]={7,7};
			period = {2,1};
			test();
			/*
				C	C	0		0
					H	1.8		0.2

				H	C	-1.8	-0.2
					H	0		0

				-1	-1	|	-1	0
				-1	0	|	-1	0
				-1	1	|	-1	0
				0	-1	|	0	-1
				0	0	|	0	-1
				0	1	|	0	-1
				1	-1	|	-1	0
				1	0	|	-1	0
				1	1	|	-1	0
				2	-1	|	0	-1
				2	0	|	0	-1
				2	1	|	0	-1
			*/
		}

		test_direction();
	}

	static void test_direction()
	{
		using TA = std::string;
		using Tcell = int;
		constexpr int Ndim = 3;
		using Tpos = double;
		constexpr int Npos = 3;
		using TC = std::array<Tcell,Ndim>;
		using Tatom_pos = std::array<Tpos,Npos>;

		std::map<TA,Tatom_pos> atoms_pos;
		std::array<Tatom_pos,Ndim> latvec;
		std::array<Tcell,Ndim> period;

		// graphene-like hexagonal lattice, period=5 in a,b directions
		latvec[0] = {10, 0, 0};
		latvec[1] = {5, std::sqrt(75.0), 0};
		latvec[2] = {0, 0, 100};
		period = {5, 5, 1};

		// both atoms at origin -> Ryx = {0,0,0}
		atoms_pos["H"] = {0, 0, 0};
		atoms_pos["C"] = {0, 0, 0};

		RI::Cell_Nearest<TA, Tcell, Ndim, Tpos, Npos> cn;
		cn.init(atoms_pos, latvec, period);
		std::cout<<cn.cells_nearest_continuous<<std::endl;

		auto run = [&](const std::string &label, const TA &Ax, const TA &Ay, const TC &cell)
		{
			double dist;
			TC result = cn.cell_nearest_direction(Ax, Ay, cell, dist);
			std::cout<<label<<": cell_nearest=("
			         <<result[0]<<","<<result[1]<<","<<result[2]
			         <<") dist="<<dist<<std::endl;
		};

		// Test A: cell(2,2,0)
		//   2 cells tie at dist = 10*sqrt(7) ~ 26.4575:
		//     {-3,2,0} (a=-1,b=0) and {2,-3,0} (a=0,b=-1)
		//   Tie-breaker: first min {-3,2,0} (|b|=0), then {2,-3,0} (|b|=1) loses
		{
			run("Test A: cell( 2, 2,0)", "H", "C", {2,2,0});
			// Expected: (-3,2,0) dist ~ 26.4575
		}

		// Test B: cell(-2,-2,0)
		//   2 cells tie at dist = 10*sqrt(7):
		//     {-2,3,0} (a=0,b=1) and {3,-2,0} (a=1,b=0)
		//   Tie-breaker: first min {-2,3,0}, then {3,-2,0} wins via |b|=0 < |b|=1
		{
			run("Test B: cell(-2,-2,0)", "H", "C", {-2,-2,0});
			// Expected: (3,-2,0) dist ~ 26.4575
		}

		// Test C: cell(3,4,0)
		//   2 cells tie at dist = 10*sqrt(7):
		//     (-2,-1,0) (a=-1,b=-1) and (3,-1,0) (a=0,b=-1)
		//   |b| both=1, then |a|=0 < |a|=1 -> switch to (3,-1,0)
		{
			run("Test C: cell( 3, 4,0)", "H", "C", {3,4,0});
			// Expected: (3,-1,0) dist ~ 26.4575
		}

		// Test D: cell(4,3,0)
		//   2 cells tie at dist = 10*sqrt(7):
		//     (-1,-2,0) (a=-1,b=-1) and (-1,3,0) (a=-1,b=0)
		//   |b|=0 < |b|=1 -> switch to (-1,3,0)
		{
			run("Test D: cell( 4, 3,0)", "H", "C", {4,3,0});
			// Expected: (-1,3,0) dist ~ 26.4575
		}

		// Test E: cell(6,7,0)
		//   2 cells tie at dist = 10*sqrt(7):
		//     (1,-3,0) (a=-1,b=-2) and (1,2,0) (a=-1,b=-1)
		//   |b|=1 < |b|=2 -> switch to (1,2,0)
		{
			run("Test E: cell( 6, 7,0)", "H", "C", {6,7,0});
			// Expected: (1,2,0) dist ~ 26.4575
		}

		// Test F: cell(7,6,0)
		//   2 cells tie at dist = 10*sqrt(7):
		//     (-3,1,0) (a=-2,b=-1) and (2,1,0) (a=-1,b=-1)
		//   |b| both=1, then |a|=1 < |a|=2 -> switch to (2,1,0)
		{
			run("Test F: cell( 7, 6,0)", "H", "C", {7,6,0});
			// Expected: (2,1,0) dist ~ 26.4575
		}
		/*
			C	C	0		0		0
				H	0		0		0
			H	C	0		0		0
				H	0		0		0

			Test A: cell( 2, 2,0): cell_nearest=(-3,2,0) dist=26.4575
			Test B: cell(-2,-2,0): cell_nearest=(3,-2,0) dist=26.4575
			Test C: cell( 3, 4,0): cell_nearest=(3,-1,0) dist=26.4575
			Test D: cell( 4, 3,0): cell_nearest=(-1,3,0) dist=26.4575
			Test E: cell( 6, 7,0): cell_nearest=(1,2,0) dist=26.4575
			Test F: cell( 7, 6,0): cell_nearest=(2,1,0) dist=26.4575
		*/
	}
}