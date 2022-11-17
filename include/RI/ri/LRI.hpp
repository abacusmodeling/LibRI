// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "LRI.h"
#include "../ri/Label.h"
#include <limits>

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
LRI<TA,Tcell,Ndim,Tdata>::LRI()
	:save_load(*this)
{
	Ds_ab.reserve(Label::array_ab.size());

	filter_funcs.reserve(Label::array_ab.size());
	for(const Label::ab &label : Label::array_ab)
		filter_funcs[label]	=
			[](const Tensor<Tdata> &D,
				const Tdata_real &threshold) -> bool
			{	return D.norm(std::numeric_limits<double>::max()) > threshold;	};

	for(std::size_t i=0; i<Ndim; ++i)
		period[i] = std::numeric_limits<Tcell>::max()/4;		// /4 for not out of range when Array_Operator::operator%

	this->set_cal_funcs_b01();
	this->set_cal_funcs_bx2();
}

}