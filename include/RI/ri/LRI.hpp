// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "LRI.h"
#include "../ri/Label.h"
#include <limits>
#include <typeinfo>

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
LRI<TA,Tcell,Ndim,Tdata>::LRI()
{
	this->data_ab_name.reserve(Label::array_ab.size());

	auto func_norm_max =
		[](const Tensor<Tdata> &D, const Tdata_real &threshold) -> bool
		{	return D.norm(std::numeric_limits<double>::max()) > threshold;	};
	this->filter_funcs.reserve(Label::array_ab.size());
	for(const Label::ab &label : Label::array_ab)
		this->filter_funcs[label] = func_norm_max;

	for(std::size_t i=0; i<Ndim; ++i)
		this->period[i] = std::numeric_limits<Tcell>::max()/4;		// /4 for not out of range when Array_Operator::operator%
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
template<typename Tfilter>
void LRI<TA,Tcell,Ndim,Tdata>::prepare_filter_atom()
{
	if (!this->flag_symmetry)
	{
		if (!this->filter_atom || typeid(*this->filter_atom) != typeid(Filter_Atom<TA,TAC>))
			this->filter_atom = std::make_shared<Filter_Atom<TA,TAC>>();
		return;
	}

	if (this->filter_atom && typeid(*this->filter_atom) == typeid(Tfilter))
	{
		const auto &filter = static_cast<const Tfilter &>(*this->filter_atom);
		// A copied LRI must bind the filter to its own configuration.
		if (&filter.symmetry.period == &this->period
			&& &filter.symmetry.irreducible_sector_ == &this->irreducible_sector)
			return;
	}
	this->filter_atom = std::make_shared<Tfilter>(this->period, this->irreducible_sector);
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void LRI<TA,Tcell,Ndim,Tdata>::cal_loop3(
	const std::vector<Label::ab_ab> &labels,
	std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_result,
	const Tdata fac_add_Ds)
{
	if(this->cal_mode == LRI_Cal_Mode::CPU)
		this->cal_loop3_CPU(labels, Ds_result, fac_add_Ds);
	else if(this->cal_mode == LRI_Cal_Mode::CPU_fine_grained_lock)
		this->cal_loop3_CPU_fine_grained_lock(labels, Ds_result, fac_add_Ds);
  #ifdef __GPU_RI
	else if(this->cal_mode == LRI_Cal_Mode::GPU)
		this->cal_loop3_GPU(labels, Ds_result, fac_add_Ds);
  #endif
	else
		throw std::invalid_argument("LRI cal_mode cannot be " + std::to_string(static_cast<std::underlying_type<LRI_Cal_Mode>::type>(this->cal_mode)));
}

}