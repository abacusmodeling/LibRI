// ==========================================
//  Author: Peize Lin, Rong Shi, Minye Zhang
//  Date:   2022.07.25
// ==========================================

#pragma once

#include "RPA.h"
#include "../ri/Label.h"
#include "../global/Map_Operator.h"
#include "./symmetry/Filter_Atom_Symmetry.h"

#include <cassert>

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void RPA<TA,Tcell,Ndim,Tdata>::set_parallel(
	const MPI_Comm &mpi_comm,
	const std::map<TA,Tatom_pos> &atoms_pos,
	const std::array<Tatom_pos,Ndim> &latvec,
	const std::array<Tcell,Ndim> &period)
{
	this->lri.set_parallel(
		mpi_comm, atoms_pos, latvec, period,
		{Label::ab_ab::a1b1_a2b2, Label::ab_ab::a1b2_a2b1});
	this->flag_finish.stru = true;
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void RPA<TA,Tcell,Ndim,Tdata>::set_symmetry(
	const bool flag_symmetry,
	const std::map<std::pair<TA,TA>, std::set<TC>> &irreducible_sector)
{
	if(flag_symmetry)
		this->lri.filter_atom = std::make_shared<Filter_Atom_Symmetry<TA,TC,Tdata>>(
			this->period, irreducible_sector);
	else
		this->lri.filter_atom = std::make_shared<Filter_Atom<TA,TAC>>();
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void RPA<TA,Tcell,Ndim,Tdata>::set_Cs(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Cs,
	const Tdata_real &threshold_C)
{
	this->lri.set_tensors_map2(
		Cs,
		{Label::ab::a, Label::ab::b},
		{{"threshold_filter", threshold_C}} );
	this->flag_finish.C = true;
}


template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
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
		this->lri.set_tensors_map2(
			Gs,
			{Label::ab::a1b1, Label::ab::a1b2},
			{{"threshold_filter", threshold_G}} );
	};

	auto set_Gs_a2 = [this, &threshold_G](const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Gs)
	{
		this->lri.set_tensors_map2(
			Gs,
			{Label::ab::a2b1, Label::ab::a2b2},
			{{"threshold_filter", threshold_G}} );
	};

	this->chi0s.clear();

	set_Gs_a1(Gs_tau_positive);
	set_Gs_a2(Gs_tau_negative);
	this->lri.cal_loop3({
		Label::ab_ab::a1b1_a2b2,
		Label::ab_ab::a1b2_a2b1},
		this->chi0s);

	set_Gs_a1(Gs_tau_negative);			// tmp
	set_Gs_a2(Gs_tau_positive);			// tmp
	//set_Gs_a1(conj(Gs_tau_negative));
	//set_Gs_a2(conj(Gs_tau_positive));
	this->lri.cal_loop3({
		Label::ab_ab::a1b1_a2b2,
		Label::ab_ab::a1b2_a2b1},
		this->chi0s);
}

}