#pragma once

#include "RI/physics/LR.h"
#include "Test_Helpers.hpp"

namespace LR_Test
{

using namespace Test_Helpers;
static const std::string ref_path = "unittests/physics/LR-ref";

// -------- main test --------
static void main(int argc, char *argv[])
{
	int mpi_init_provide;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_init_provide);

	// ====== config: 1 atom, 3x1x1 k-grid, nocc=1, nvirt=2, nabf=2, nw=3 ======
	const TA atom0 = 0;
	const int nk = 3;
	const int nocc = 1, nvirt = 2, nband = nocc + nvirt;
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

	// Vs / Ws: shape {nabf=2, nabf=2}, values vary with R
	auto make_V = [&]() {
		std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> V;
		for (auto R : {std::array<int,3>{0,0,0}, std::array<int,3>{1,0,0}, std::array<int,3>{-1,0,0}}) {
			const double w = (R[0]==0 && R[1]==0 && R[2]==0) ? 0.01 : 0.005 / (std::abs(R[0])+std::abs(R[1])+std::abs(R[2]));
			V[atom0][{atom0, R}] = RI::Tensor<Tdata>({nabf, nabf});
			for (int i = 0; i < nabf; ++i)
				for (int j = 0; j < nabf; ++j)
					V[atom0][{atom0, R}](i,j) = (i==j) ? Tdata(w, 0.0) : Tdata(w*0.5, 0.0);
		}
		return V;
	};
	auto Vs = make_V();
	auto Ws = make_V();

	// ====== init LR with MPI distribution ======
	RI::LR<TA,Tcell,Ndim,Tdata> lr;
	lr.init(kfrac, nocc, nvirt);
	lr.set_parallel(MPI_COMM_WORLD, 1, nk, {{1,1,1}});
	lr.set_Cs(Cs, 0.0, {atom0}, {atom0});
	lr.set_Vs(Vs, 0.0, {atom0}, {atom0});
	lr.set_Ws(Ws, 0.0, {atom0}, {atom0});

	// build q_list / q2kpair from distributed k1, k2
	{
		auto mod1 = [](double x) { double r = x - std::floor(x); return r < 0 ? r+1.0 : r; };
		for (int k1 : lr.k1_indices)
			for (int k2 : lr.k2_indices) {
				Tk q = {{mod1(kfrac[k2][0]-kfrac[k1][0]),
				         mod1(kfrac[k2][1]-kfrac[k1][1]),
				         mod1(kfrac[k2][2]-kfrac[k1][2])}};
				lr.q2kpair[q].emplace_back(k1, k2);
			}
		for (auto& p : lr.q2kpair) lr.q_list.push_back(p.first);
	}

	// map_psi: only for distributed k_indices
	for (int ik : lr.k_indices) {
		lr.map_psi[ik][atom0] = RI::Tensor<Tdata>({nband, nw});
		for (int m = 0; m < nband; ++m)
			for (int t = 0; t < nw; ++t)
				lr.map_psi[ik][atom0](m,t)
					= Tdata(0.1 * (m+1.0)*(t+1.0)*(ik+1.0),
					        0.1 * (m+0.5)*(t+0.5)*(ik+0.5));
	}

	// compute all 5 results
	std::ofstream ofs("/dev/null");
	lr.cal_Csk_ao_mo("Cs_", ofs);
	auto V_A = lr.cal_cvc_mo_k_hartree_onthefly({"O","V","O","V"}, "Vs_", true);
	auto V_B = lr.cal_cvc_mo_k_hartree_onthefly({"O","V","O","V"}, "Vs_", false);
	auto W_A = lr.cal_cvc_mo_k_onthefly({"O","O","V","V"}, "Ws_", true);
	auto W_B = lr.cal_cvc_mo_k_onthefly({"V","O","O","V"}, "Ws_", false);

	// ====== gather per-rank results to rank 0, then regression ======
	int nproc; MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	auto Csk_gathered = gather_map(lr.Csk_ao_mo, 0, MPI_COMM_WORLD, 0);
	auto VA_gathered   = gather_map(V_A, 0, MPI_COMM_WORLD, 1);
	auto VB_gathered   = gather_map(V_B, 0, MPI_COMM_WORLD, 2);
	auto WA_gathered   = gather_map(W_A, 0, MPI_COMM_WORLD, 3);
	auto WB_gathered   = gather_map(W_B, 0, MPI_COMM_WORLD, 4);

	if (nproc > 1) MPI_Barrier(MPI_COMM_WORLD);

	if (argc >= 2 && std::string(argv[1]) == "--write-ref")
	{
		assert(nproc == 1 && "--write-ref requires exactly 1 MPI process");
		std::ofstream ofs_ref(ref_path);
		assert(ofs_ref && "Failed to open ref file for writing");
		ofs_ref << std::scientific << std::setprecision(REF_PRECISION);
		write_map(ofs_ref, Csk_gathered);
		write_map(ofs_ref, VA_gathered);
		write_map(ofs_ref, VB_gathered);
		write_map(ofs_ref, WA_gathered);
		write_map(ofs_ref, WB_gathered);
		std::cout << "LR_ref written to " << ref_path << std::endl;
	}
	else
	{
		int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (rank == 0)
		{
			std::ifstream ifs_ref(ref_path);
			assert(ifs_ref && "LR_ref not found; run with --write-ref to generate");
			TensorMap ref;
			read_map(ifs_ref, ref); compare_map(Csk_gathered, ref);
			read_map(ifs_ref, ref); compare_map(VA_gathered, ref);
			read_map(ifs_ref, ref); compare_map(VB_gathered, ref);
			read_map(ifs_ref, ref); compare_map(WA_gathered, ref);
			read_map(ifs_ref, ref); compare_map(WB_gathered, ref);
		}
	}

	lr.free_Cs();
	lr.free_Vs();
	lr.free_Ws();

	MPI_Finalize();
}

}
