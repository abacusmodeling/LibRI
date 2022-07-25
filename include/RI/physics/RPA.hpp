// ==========================================
//  Author: Peize Lin, Rong Shi, Minye Zhang
//  Date:   2022.07.25
// ==========================================

#pragma once

#include "RPA.h"
#include "../ri/Label.h"
#include "../global/Map_Operator.h"

#include <cassert>

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void RPA<TA,Tcell,Ndim,Tdata>::set_stru(
	const std::map<TA,TatomR> &atomsR,
	const std::array<TatomR,Ndim> &latvec,
	const std::array<Tcell,Ndim> &period)
{
	this->lri.set_parallel(atomsR, latvec, period);
	this->flag_finish.stru = true;
}

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void RPA<TA,Tcell,Ndim,Tdata>::set_Cs(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Cs,
	const Tdata_real &threshold_C)
{
	this->lri.set_tensors_map2( Cs, Label::ab::a, threshold_C );
	this->lri.set_tensors_map2( Cs, Label::ab::b, threshold_C );
	this->flag_finish.C = true;
}


template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void RPA<TA,Tcell,Ndim,Tdata>::cal_chi0s(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Gs_tau_positive,
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Gs_tau_negative,
	const Tdata_real &threshold_G)
{
	assert(this->flag_finish.stru);
	assert(this->flag_finish.C);

	using namespace Map_Operator;

	auto set_Gs_a1 = [this, &threshold_G](const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Gs)
	{
		this->lri.set_tensors_map2( Gs, Label::ab::a1b1, threshold_G );
		this->lri.set_tensors_map2( Gs, Label::ab::a1b2, threshold_G );
	};

	auto set_Gs_a2 = [this, &threshold_G](const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Gs)
	{
		this->lri.set_tensors_map2( Gs, Label::ab::a2b1, threshold_G );
		this->lri.set_tensors_map2( Gs, Label::ab::a2b2, threshold_G );
	};

	set_Gs_a1(Gs_tau_positive);
	set_Gs_a2(Gs_tau_negative);
	this->chi0s = this->lri.cal({
		Label::ab_ab::a1b1_a2b2,
		Label::ab_ab::a1b2_a2b1});

	set_Gs_a1(Gs_tau_negative);			// tmp
	set_Gs_a2(Gs_tau_positive);			// tmp
	//set_Gs_a1(conj(Gs_tau_negative));
	//set_Gs_a2(conj(Gs_tau_positive));
	this->chi0s = this->chi0s + this->lri.cal({
		Label::ab_ab::a1b1_a2b2,
		Label::ab_ab::a1b2_a2b1});
}

