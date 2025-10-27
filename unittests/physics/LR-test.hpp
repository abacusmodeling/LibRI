#pragma once

#define LR_REF_PRECISION 9   // significant digits when writing reference file
#define LR_REF_TOLERANCE 1e-7 // tolerance when comparing against reference

#include "RI/physics/LR.h"
#include <cassert>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace LR_Test
{

using Tdata = std::complex<double>;
using TA = int;
using Tcell = int;
constexpr int Ndim = 3;
using TAC = std::pair<TA,std::array<Tcell,Ndim>>;
using Tk = std::array<double,Ndim>;

static const std::string ref_path = "unittests/physics/LR-ref";

// -------- helpers: serialize / deserialize 2-level maps of arbitrary-rank tensors --------
using TensorMap = std::map<int, std::map<int, RI::Tensor<Tdata>>>;// map<k1, map<k2, Tensor>> or map<k, map<atom, Tensor>>

static void write_tensor_map(std::ofstream &ofs, const TensorMap &m)
{
	ofs << m.size() << "\n";
	for (auto &[k1, inner] : m) {
		ofs << k1 << " " << inner.size() << "\n";
		for (auto &[k2, t] : inner) {
			ofs << k2 << " " << t.shape.size();
			for (auto d : t.shape) ofs << " " << d;
			std::size_t total = 1;
			for (auto d : t.shape) total *= d;
			for (std::size_t i = 0; i < total; ++i)
				ofs << " " << std::setprecision(LR_REF_PRECISION) << std::scientific
				    << t.ptr()[i].real() << " " << t.ptr()[i].imag();
			ofs << "\n";
		}
	}
}

static void read_tensor_map(std::ifstream &ifs, TensorMap &m)
{
	m.clear();
	std::size_t n1; ifs >> n1;
	for (std::size_t i = 0; i < n1; ++i) {
		int k1; std::size_t n2;
		ifs >> k1 >> n2;
		for (std::size_t j = 0; j < n2; ++j) {
			int k2; std::size_t ndim;
			ifs >> k2 >> ndim;
			std::vector<std::size_t> shape(ndim);
			std::size_t total = 1;
			for (std::size_t d = 0; d < ndim; ++d) {
				ifs >> shape[d]; total *= shape[d];
			}
			RI::Tensor<Tdata> t(shape);
			for (std::size_t p = 0; p < total; ++p) {
				double re, im; ifs >> re >> im; t.ptr()[p] = Tdata(re, im);
			}
			m[k1][k2] = std::move(t);
		}
	}
}

static void compare_tensor_map(const TensorMap &a, const TensorMap &b)
{
	assert(a.size() == b.size());
	for (auto &[k1, inner_a] : a) {
		assert(b.count(k1));
		auto &inner_b = b.at(k1);
		assert(inner_a.size() == inner_b.size());
		for (auto &[k2, t_a] : inner_a) {
			assert(inner_b.count(k2));
			auto &t_b = inner_b.at(k2);
			assert(t_a.shape.size() == t_b.shape.size());
			std::size_t total = 1;
			for (std::size_t d = 0; d < t_a.shape.size(); ++d) {
				assert(t_a.shape[d] == t_b.shape[d]);
				total *= t_a.shape[d];
			}
			for (std::size_t p = 0; p < total; ++p)
				assert(std::abs(t_a.ptr()[p] - t_b.ptr()[p]) < LR_REF_TOLERANCE);
		}
	}
}

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

	// k1 = k2 = k_indices = all 3
	std::vector<int> k_indices(nk), k1_indices(nk), k2_indices(nk);
	for (int i = 0; i < nk; ++i)
		k_indices[i] = k1_indices[i] = k2_indices[i] = i;

	// build q_list / q2kpair from k1-k2 differences
	std::vector<Tk> q_list;
	std::map<Tk, std::vector<std::pair<int,int>>> q2kpair;
	{
		auto mod1 = [](double x) { double r = x - std::floor(x); return r < 0 ? r+1.0 : r; };
		for (int k1 : k1_indices)
			for (int k2 : k2_indices) {
				Tk q = {{mod1(kfrac[k2][0]-kfrac[k1][0]),
				         mod1(kfrac[k2][1]-kfrac[k1][1]),
				         mod1(kfrac[k2][2]-kfrac[k1][2])}};
				q2kpair[q].emplace_back(k1, k2);
			}
		for (auto& p : q2kpair) q_list.push_back(p.first);
	}

	// Cs: shape {nabf=2, nw=3, nw=3}, all 1.0+0i
	std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Cs;
	Cs[atom0][{atom0, {{0,0,0}}}] = RI::Tensor<Tdata>({nabf, nw, nw});
	for (int mu = 0; mu < nabf; ++mu)
		for (int s = 0; s < nw; ++s)
			for (int t = 0; t < nw; ++t)
				Cs[atom0][{atom0, {{0,0,0}}}](mu,s,t) = Tdata(1.0, 0.0);

	// Vs / Ws: shape {nabf=2, nabf=2}, diag=0.01, off-diag=0.005
	auto make_V = [&]() {
		std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> V;
		V[atom0][{atom0, {{0,0,0}}}] = RI::Tensor<Tdata>({nabf, nabf});
		for (int i = 0; i < nabf; ++i)
			for (int j = 0; j < nabf; ++j)
				V[atom0][{atom0, {{0,0,0}}}](i,j) = (i==j) ? Tdata(0.01, 0.0) : Tdata(0.005, 0.0);
		return V;
	};
	auto Vs = make_V();
	auto Ws = make_V();

	// map_psi: for each k, shape {nband=3, nw=3}, complex values
	std::map<int, std::map<TA, RI::Tensor<Tdata>>> map_psi;
	for (int ik : k_indices) {
		map_psi[ik][atom0] = RI::Tensor<Tdata>({nband, nw});
		for (int m = 0; m < nband; ++m)
			for (int t = 0; t < nw; ++t)
				map_psi[ik][atom0](m,t)
					= Tdata(0.1 * (m+1.0)*(t+1.0)*(ik+1.0),
					        0.1 * (m+0.5)*(t+0.5)*(ik+0.5));
	}

	// ====== init LR ======
	RI::LR<TA,Tcell,Ndim,Tdata> lr;
	std::array<std::array<double,3>,3> latvec = {{{{1,0,0}}, {{0,1,0}}, {{0,0,1}}}};
	lr.set_parallel(MPI_COMM_WORLD, {{{atom0, {{0,0,0}}}}}, latvec, {{1,1,1}});
	lr.set_Cs(Cs, 0.0, {atom0}, {atom0});
	lr.set_Vs(Vs, 0.0, {atom0}, {atom0});
	lr.set_Ws(Ws, 0.0, {atom0}, {atom0});

	lr.k1_indices = k1_indices;
	lr.k2_indices = k2_indices;
	lr.k_indices  = k_indices;
	lr.list_I  = {atom0}; lr.list_J  = {atom0}; lr.list_IJ = {atom0};
	lr.kindex_map = kfrac;
	lr.nocc = nocc; lr.nvirt = nvirt;
	lr.map_psi = std::move(map_psi);
	lr.q_list  = q_list; lr.q2kpair = q2kpair;

	// compute all 5 results
	std::ofstream ofs("/dev/null");
	lr.cal_Csk_ao_mo("Cs_", ofs);
	auto V_A = lr.cal_cvc_mo_k_hartree_onthefly({"O","V","O","V"}, "Vs_", true);
	auto V_B = lr.cal_cvc_mo_k_hartree_onthefly({"O","V","O","V"}, "Vs_", false);
	auto W_A = lr.cal_cvc_mo_k_onthefly({"O","O","V","V"}, "Ws_", true);
	auto W_B = lr.cal_cvc_mo_k_onthefly({"V","O","O","V"}, "Ws_", false);

	// ====== regression: write or compare reference file ======
	if (argc >= 2 && std::string(argv[1]) == "--write-ref")
	{
		std::ofstream ofs_ref(ref_path);
		assert(ofs_ref && "Failed to open ref file for writing");
		ofs_ref << std::scientific << std::setprecision(LR_REF_PRECISION);
		write_tensor_map(ofs_ref, lr.Csk_ao_mo);
		write_tensor_map(ofs_ref, V_A);
		write_tensor_map(ofs_ref, V_B);
		write_tensor_map(ofs_ref, W_A);
		write_tensor_map(ofs_ref, W_B);
		std::cout << "LR_ref written to " << ref_path << std::endl;
	}
	else
	{
		std::ifstream ifs_ref(ref_path);
		assert(ifs_ref && "LR_ref not found; run with --write-ref to generate");
		TensorMap ref;
		read_tensor_map(ifs_ref, ref); compare_tensor_map(lr.Csk_ao_mo, ref);
		read_tensor_map(ifs_ref, ref); compare_tensor_map(V_A, ref);
		read_tensor_map(ifs_ref, ref); compare_tensor_map(V_B, ref);
		read_tensor_map(ifs_ref, ref); compare_tensor_map(W_A, ref);
		read_tensor_map(ifs_ref, ref); compare_tensor_map(W_B, ref);
	}

	lr.free_Cs();
	lr.free_Vs();
	lr.free_Ws();

	MPI_Finalize();
}

}