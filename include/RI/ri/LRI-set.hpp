// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "LRI.h"
#include "RI_Tools.h"
#include "CS_Matrix.h"
#include "../comm/mix/Communicate_Tensors.h"
#include <algorithm>

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void LRI<TA,Tcell,Ndim,Tdata>::set_tensors_map2(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_local,
	const Label::ab &label,
	const Global_Func::To_Real_t<Tdata> &threshold)
{
	//if()
		std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_period = RI_Tools::cal_period(Ds_local, this->period);
	std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_comm = this->comm_tensors_map2(label, std::move(Ds_period));

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

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
std::map<TA,std::map<std::pair<TA,std::array<Tcell,Ndim>>,Tensor<Tdata>>>
LRI<TA,Tcell,Ndim,Tdata>::comm_tensors_map2(
	const Label::ab &label,
	const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds) const
{
	switch(label)
	{
		case Label::ab::a:
			return Communicate_Tensors::comm_judge_map2(this->mpi_comm, Ds, this->list_Aa01(), this->list_Aa2());
		case Label::ab::b:
			return Communicate_Tensors::comm_judge_map2_period(this->mpi_comm, Ds, this->list_Ab01(), this->list_Ab2(), this->period);
		case Label::ab::a0b0:	case Label::ab::a0b1:
		case Label::ab::a1b0:	case Label::ab::a1b1:
			return Communicate_Tensors::comm_judge_map2(this->mpi_comm, Ds, this->list_Aa01(), this->list_Ab01());
		case Label::ab::a0b2:
		case Label::ab::a1b2:
			return Communicate_Tensors::comm_judge_map2(this->mpi_comm, Ds, this->list_Aa01(), this->list_Ab2());
		case Label::ab::a2b0:	case Label::ab::a2b1:
			return Communicate_Tensors::comm_judge_map2_period(this->mpi_comm, Ds, this->list_Aa2(), this->list_Ab01(), this->period);
		case Label::ab::a2b2:
			return Communicate_Tensors::comm_judge_map2_period(this->mpi_comm, Ds, this->list_Aa2(), this->list_Ab2(), this->period);
		default:
			throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
	}
}

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