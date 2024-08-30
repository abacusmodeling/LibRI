// ==========================================
//  Author: Peize Lin, Rong Shi, Minye Zhang
//  Date:   2022.07.25
// ==========================================

#pragma once

#include "../ri/LRI.h"
#include "../global/Tensor.h"
#include "../global/Global_Func-2.h"

#include <mpi.h>
#include <array>
#include <map>
#include <set>

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
class RPA
{
public:
	using TC = std::array<Tcell,Ndim>;
	using TAC = std::pair<TA,TC>;
	using Tdata_real = Global_Func::To_Real_t<Tdata>;
	using Tatom_pos = std::array<double,Ndim>;		// tmp

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

	void cal_chi0s(
		const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Gs_tau_positive,
		const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Gs_tau_negative,
		const Tdata_real &threshold_G);

	std::map<TA, std::map<TAC, Tensor<Tdata>>> chi0s;

public:
	LRI<TA,Tcell,Ndim,Tdata> lri;

	struct Flag_Finish
	{
		bool stru=false;
		bool C=false;
	};
	Flag_Finish flag_finish;
};

}

#include "RPA.hpp"