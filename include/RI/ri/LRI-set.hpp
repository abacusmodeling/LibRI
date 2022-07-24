// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "LRI.h"
#include "RI_Tools.h"
#include "CS_Matrix.h"
#include <algorithm>

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void LRI<TA,Tcell,Ndim,Tdata>::set_parallel(
	const std::map<TA,TatomR> &atomsR,
	const std::array<TatomR,Ndim> &latvec,
	const std::array<Tcell,Ndim> &period_in)
{
	this->parallel->set_parallel( this->mpi_comm, atomsR, latvec, period_in );
	this->period = period_in;
}

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void LRI<TA,Tcell,Ndim,Tdata>::set_tensors_map2(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_local,
	const Label::ab &label,
	const Global_Func::To_Real_t<Tdata> &threshold)
{
	//if()
		std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_period = RI_Tools::cal_period(Ds_local, this->period);

	std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_comm = this->parallel->comm_tensors_map2(label, std::move(Ds_period));

	if(threshold)
		this->Ds_ab[label] = RI_Tools::filter(std::move(Ds_comm), filter_funcs[label], threshold);
	else
		this->Ds_ab[label] = std::move(Ds_comm);

	if(this->csm.threshold_max)
		this->csm.set_tensor(label, this->Ds_ab[label]);
}

/*
template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void RI<TA,Tcell,Ndim,Tdata>::set_tensors_map3(
	const std::map<TA, std::map<TAC, std::map<TAC, Tensor<Tdata>>>> &Ds_local,
	const Label::ab &label,
	const Global_Func::To_Real_t<Tdata> &threshold)
{
	if(label==Label::ab::a)
	{
		//this->Ds_ab[label] = Communicate::communicate(Ds_ab, threshold);
		if(threshold)
			this->Ds_a = LRI::filter(Ds_local, filter_func, threshold);
		else
			this->Ds_a = Ds_local;
		//this->cs_matrix.set_tensor2(this->Ds_ab[label], label);
	}
	else if(label==Label::ab::b)
	{
		//this->Ds_ab[label] = Communicate::communicate(Ds_ab, threshold);
		if(threshold)
			this->Ds_b = LRI::filter(Ds_local, filter_func, threshold);
		else
			this->Ds_b = Ds_local;
		//this->cs_matrix.set_tensor2(this->Ds_ab[label], label);
	}
}
*/


/*
template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
std::map<TA,std::map<TAC,std::map<TAC,Tensor<Tdata>>>>
RI<TA,Tcell,Ndim,Tdata>::comm_tensors_map3(
	const Label::ab &label,
	const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds) const
{
	switch(label)
	{
		case Label::ab:a:
			return Map3(this->mpi_comm, Ds, this->list_Aa0(), this->list_Aa1(), this->list_Aa2());
		case Label::ab:b:
			return Map3_period(this->mpi_comm, Ds, this->list_Ab0(), this->list_Ab1(), this->list_Ab2());
		default:
			throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
	}
}
*/