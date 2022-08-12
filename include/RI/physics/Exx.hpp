// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "Exx.h"
#include "../ri/Label.h"

#include <cassert>

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::set_parallel(
	const MPI_Comm &mpi_comm,
	const std::map<TA,TatomR> &atomsR,
	const std::array<TatomR,Ndim> &latvec,
	const std::array<Tcell,Ndim> &period)
{
	this->lri.set_parallel(mpi_comm, atomsR, latvec, period);
	this->flag_finish.stru = true;
}

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::set_Cs(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Cs,
	const Tdata_real &threshold_C)
{
	this->lri.set_tensors_map2( Cs, Label::ab::a, threshold_C );
	this->lri.set_tensors_map2( Cs, Label::ab::b, threshold_C );
	this->flag_finish.C = true;
}

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::set_Vs(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Vs,
	const Tdata_real &threshold_V)
{
	this->lri.set_tensors_map2( Vs, Label::ab::a0b0, threshold_V );
	this->flag_finish.V = true;
}

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::set_Ds(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds,
	const Tdata_real &threshold_D)
{
	this->lri.set_tensors_map2( Ds, Label::ab::a1b1, threshold_D );
	this->lri.set_tensors_map2( Ds, Label::ab::a1b2, threshold_D );
	this->lri.set_tensors_map2( Ds, Label::ab::a2b1, threshold_D );
	this->lri.set_tensors_map2( Ds, Label::ab::a2b2, threshold_D );
	this->flag_finish.D = true;
}

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::cal_Hs()
{
	assert(this->flag_finish.stru);
	assert(this->flag_finish.C);
	assert(this->flag_finish.V);
	assert(this->flag_finish.D);

	this->Hs = this->lri.cal({
		Label::ab_ab::a0b0_a1b1,
		Label::ab_ab::a0b0_a1b2,
		Label::ab_ab::a0b0_a2b1,
		Label::ab_ab::a0b0_a2b2});
}