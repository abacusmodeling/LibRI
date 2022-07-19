// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "LRI.h"
#include "RI_Tools.h"
#include "CS_Matrix.h"
#include <algorithm>


template<typename TA, typename Tperiod, size_t Ndim_period, typename Tdata>
void LRI<TA,Tperiod,Ndim_period,Tdata>::set_tensor2(
	const std::map<TA, std::map<TAp, Tensor<Tdata>>> &Ds_local,
	const Label::ab &label,
	const Global_Func::To_Real_t<Tdata> &threshold)
{
	//if()
		std::map<TA, std::map<TAp, Tensor<Tdata>>> Ds_period = RI_Tools::cal_period(Ds_local, this->period);
	//this->Ds_ab[label] = Communicate::communicate(Ds_period, threshold);
	if(threshold)
		this->Ds_ab[label] = RI_Tools::filter(Ds_period, filter_funcs[label], threshold);
	else
		this->Ds_ab[label] = std::move(Ds_period);
	if(this->csm.threshold_max)
		this->csm.set_tensor(label, this->Ds_ab[label]);
}

/*
void LRI::set_tensor3(
	const std::map<TA, std::map<TAp, std::map<TAp, Tensor<Tdata>>>> &Ds_local,
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
