// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "LRI.h"
#include "Label.h"
#include <limits>

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
LRI<TA,Tcell,Ndim,Tdata>::LRI()
{
	Ds_ab.reserve(Label::array_ab.size());

	filter_funcs.reserve(Label::array_ab.size());
	for(const Label::ab &label : Label::array_ab)
		filter_funcs[label]	=
			[](const Tensor<Tdata> &D,
				const Global_Func::To_Real_t<Tdata> &thr) -> bool
			{	return D.norm(std::numeric_limits<double>::max()) > thr;	};

	for(size_t i=0; i<Ndim; ++i)
		period[i] = std::numeric_limits<Tcell>::max();
}