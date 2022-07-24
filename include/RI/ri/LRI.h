// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "../ri/Label.h"
#include "../global/Tensor.h"
#include "CS_Matrix.h"
#include "../parallel/Parallel_LRI_Equally.h"
#include "RI_Tools.h"
#include "../global/Global_Func-2.h"

#include <mpi.h>
#include <map>
#include <set>
#include <array>
#include <unordered_map>
#include <memory>
#include <functional>

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
class LRI
{
public:
	using TC = std::array<Tcell,Ndim>;
	using TAC = std::pair<TA,TC>;
	using TatomR = std::array<double,Ndim>;		// tmp

	std::unordered_map<Label::ab, std::map<TA, std::map<TAC, Tensor<Tdata>>>> Ds_ab;
	std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_result;

	LRI(const MPI_Comm &mpi_comm_in);

	//template<typename TatomR>
	void set_parallel(
		const std::map<TA,TatomR> &atomsR,
		const std::array<TatomR,Ndim> &latvec,
		const std::array<Tcell,Ndim> &period_in);	

	void set_tensors_map2(
		const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_local,
		const Label::ab &label,
		const Global_Func::To_Real_t<Tdata> &threshold);
	//void set_tensors_map3(
	//	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_local,
	//	const std::string &label,
	//	const Tdata &threshold);

	void cal(const std::vector<Label::ab_ab> &lables);

	CS_Matrix< TA,TA,Ndim,
		Global_Func::To_Real_t<Tdata> > csm;
	std::shared_ptr<Parallel_LRI<TA,Tcell,Ndim,Tdata>>
		parallel = std::make_shared<Parallel_LRI_Equally<TA,Tcell,Ndim,Tdata>>();
	std::unordered_map<	Label::ab, RI_Tools::T_filter_func<Tdata> > filter_funcs;

// private:
public:
	std::array<Tcell,Ndim> period;
	const MPI_Comm mpi_comm;
};

#include "LRI.hpp"
#include "LRI-set.hpp"
#include "LRI-cal.hpp"