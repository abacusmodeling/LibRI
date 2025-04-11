// ===================
//  Author: Minye Zhang, almost completely copied from Exx.h
//  date: 2022.12.18
// ===================

#pragma once

// #include "Exx_Post_2D.h"
#include "../global/Global_Func-2.h"
#include "../global/Tensor.h"
#include "../ri/LRI.h"

#include <mpi.h>
#include <array>
#include <map>
#include <set>

namespace RI
{

//! class to compute the correlation self-energy in G0W0 approximation by space-time method
template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
class G0W0
{
public:
	using TC = std::array<Tcell,Ndim>;
	using TAC = std::pair<TA,TC>;
	using Tdata_real = Global_Func::To_Real_t<Tdata>;
	constexpr static std::size_t Npos = Ndim;		// tmp
	using Tatom_pos = std::array<double,Npos>;		// tmp

	void set_parallel(
		const MPI_Comm &mpi_comm,
		const std::map<TA,Tatom_pos> &atoms_pos,
		const std::array<Tatom_pos,Ndim> &latvec,
		const std::array<Tcell,Ndim> &period);

	void set_symmetry(
		const bool flag_symmetry,
		const std::map<std::pair<TA,TA>, std::set<TC>> &irreducible_sector);

	void set_Cs(
		const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Cs,
		const Tdata_real &threshold_C);

	// Set screened Coulomb for a particular imaginary time point (both positive and negative)
	void set_Wc(
		const std::map<TA, std::map<TAC, Tensor<Tdata>>> Wc_tau,
		const Tdata_real &threshold_Wc);

	void cal_Sigc(
		const std::map<TA, std::map<TAC, Tensor<Tdata>>> gf_tau,
		const Tdata_real &threshold_G);

	void free_G();

	// Clean up screened Coulomb for next calculation
	void free_Wc();

	std::map<TA, std::map<TAC, Tensor<Tdata>>> Sigc_tau;

public:
	LRI<TA,Tcell,Ndim,Tdata> lri;

	struct Flag_Finish
	{
		bool stru=false;
		bool C=false;
		bool W=false;
	};
	Flag_Finish flag_finish;

	MPI_Comm mpi_comm;
	std::map<TA,Tatom_pos> atoms_pos;
	std::array<Tatom_pos,Ndim> latvec;
	std::array<Tcell,Ndim> period;
};

}

#include "GW.hpp"