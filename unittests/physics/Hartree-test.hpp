// ===================
//  Author: Ziqing Guan
//  date: 2026.08.02
// ===================

#pragma once

#include "RI/physics/Hartree.h"
#include "Test_Helpers.hpp"

namespace Hartree_Test
{

using namespace Test_Helpers;
static const std::string ref_path = "unittests/physics/Hartree-ref";

// -------- main test --------
static void main(int argc, char *argv[])
{
	int mpi_init_provide;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_init_provide);

	// ====== config: 1 atom, 3x1x1 k-grid, nabf=2, nw=3 ======
	const TA atom0 = 0;
	const int nk = 3;
	const int nabf = 2, nw = 3;

	// 3x1x1 uniform k-grid fractional coordinates
	std::vector<Tk> kfrac(nk);
	for (int i = 0; i < nk; ++i)
		kfrac[i] = {{double(i)/3.0, 0.0, 0.0}};

	// Cs: shape {nabf=2, nw=3, nw=3}, values vary with R
	std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Cs;
	for (auto R : {std::array<int,3>{0,0,0}, std::array<int,3>{1,0,0}, std::array<int,3>{-1,0,0}}) {
		const double w = (R[0]==0 && R[1]==0 && R[2]==0) ? 1.0 : 0.5 / (std::abs(R[0])+std::abs(R[1])+std::abs(R[2]));
		Cs[atom0][{atom0, R}] = RI::Tensor<Tdata>({nabf, nw, nw});
		for (int mu = 0; mu < nabf; ++mu)
			for (int s = 0; s < nw; ++s)
				for (int t = 0; t < nw; ++t)
					Cs[atom0][{atom0, R}](mu,s,t) = Tdata(w * (mu+1) * (s+1) * (t+1), 0.0);
	}

	// Vs: shape {nabf=2, nabf=2}, values vary with R
	std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs;
	for (auto R : {std::array<int,3>{0,0,0}, std::array<int,3>{1,0,0}, std::array<int,3>{-1,0,0}}) {
		const double w = (R[0]==0 && R[1]==0 && R[2]==0) ? 0.01 : 0.005 / (std::abs(R[0])+std::abs(R[1])+std::abs(R[2]));
		Vs[atom0][{atom0, R}] = RI::Tensor<Tdata>({nabf, nabf});
		for (int i = 0; i < nabf; ++i)
			for (int j = 0; j < nabf; ++j)
				Vs[atom0][{atom0, R}](i,j) = (i==j) ? Tdata(w, 0.0) : Tdata(w*0.5, 0.0);
	}

	// Ds: D(s,t)[k] for all k, shape {nw=3, nw=3}, complex values
	std::map<TA, std::map<TA, std::map<int, RI::Tensor<Tdata>>>> Ds;
	for (int ik = 0; ik < nk; ++ik) {
		Ds[atom0][atom0][ik] = RI::Tensor<Tdata>({nw, nw});
		for (int i = 0; i < nw; ++i)
			for (int j = 0; j < nw; ++j)
				Ds[atom0][atom0][ik](i,j)
					= Tdata(0.1 * (i+1.0)*(j+1.0)*(ik+1.0),
					        0.1 * (i+0.5)*(j+0.5)*(ik+0.5));
	}

	// ====== init Hartree with MPI distribution ======
	RI::Hartree<TA,Tcell,Ndim,Tdata> hartree;
	hartree.init(kfrac);
	hartree.set_parallel(MPI_COMM_WORLD, 1, nk, {{3,1,1}});
	hartree.set_Cs(Cs, 0.0, {atom0}, {atom0});
	hartree.set_Vs(Vs, 0.0, {atom0}, {atom0});

	// compute
	auto result = hartree.cal_hartree(Ds);

	// ====== gather per-rank results to rank 0, then regression ======
	int nproc; MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	auto gathered = gather_map(result, 0, MPI_COMM_WORLD, 0);

	if (nproc > 1) MPI_Barrier(MPI_COMM_WORLD);

	if (argc >= 2 && std::string(argv[1]) == "--write-ref")
	{
		assert(nproc == 1 && "--write-ref requires exactly 1 MPI process");
		std::ofstream ofs_ref(ref_path);
		assert(ofs_ref && "Failed to open ref file for writing");
		ofs_ref << std::scientific << std::setprecision(REF_PRECISION);
		write_map(ofs_ref, gathered);
		std::cout << "Hartree_ref written to " << ref_path << std::endl;
	}
	else
	{
		int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (rank == 0)
		{
			std::ifstream ifs_ref(ref_path);
			assert(ifs_ref && "Hartree_ref not found; run with --write-ref to generate");
			HartreeMap ref;
			read_map(ifs_ref, ref);
			compare_map(gathered, ref);
		}
	}

	hartree.free_Cs();
	hartree.free_Vs();

	MPI_Finalize();
}

}
