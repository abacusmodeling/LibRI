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

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
class RPA
{
public:
	using TAC = std::pair<TA,std::array<Tcell,Ndim>>;
	using Tdata_real = Global_Func::To_Real_t<Tdata>;
	using TatomR = std::array<double,Ndim>;		// tmp

	RPA(const MPI_Comm &mpi_comm):lri(mpi_comm){}

	void set_stru(
		const std::map<TA,TatomR> &atomsR,
		const std::array<TatomR,Ndim> &latvec,
		const std::array<Tcell,Ndim> &period);

	void set_Cs(
		const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Cs,
		const Tdata_real &threshold_C);
	void set_csm_threshold(const Tdata_real &threshold){ this->lri.csm.set_threshold(threshold); }

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

#include "RPA.hpp"