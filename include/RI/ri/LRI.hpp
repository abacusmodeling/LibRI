// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "LRI.h"
#include "../ri/Label.h"
#include <limits>

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
LRI<TA,Tcell,Ndim,Tdata>::LRI(const MPI_Comm &mpi_comm_in)
	:mpi_comm(mpi_comm_in)
{
	Ds_ab.reserve(Label::array_ab.size());

	filter_funcs.reserve(Label::array_ab.size());
	for(const Label::ab &label : Label::array_ab)
		filter_funcs[label]	=
			[](const Tensor<Tdata> &D,
				const Tdata_real &thr) -> bool
			{	return D.norm(std::numeric_limits<double>::max()) > thr;	};

	for(size_t i=0; i<Ndim; ++i)
		period[i] = std::numeric_limits<Tcell>::max();
}