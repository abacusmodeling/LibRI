// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "LRI_Cal_Tools.h"
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
	using Tdata_real = Global_Func::To_Real_t<Tdata>;
	using TatomR = std::array<double,Ndim>;		// tmp
	
	LRI();

	//template<typename TatomR>
	void set_parallel(
		const MPI_Comm &mpi_comm_in,
		const std::map<TA,TatomR> &atomsR,
		const std::array<TatomR,Ndim> &latvec,
		const std::array<Tcell,Ndim> &period_in);	

	void set_tensors_map2(
		const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_local,
		const Label::ab &label,
		const Tdata_real &threshold);
	//void set_tensors_map3(
	//	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_local,
	//	const std::string &label,
	//	const Tdata &threshold);

	std::map<TA, std::map<TAC, Tensor<Tdata>>> cal(const std::vector<Label::ab_ab> &lables);

public:
	std::shared_ptr<Parallel_LRI<TA,Tcell,Ndim,Tdata>>
		parallel = std::make_shared<Parallel_LRI_Equally<TA,Tcell,Ndim,Tdata>>();
	std::unordered_map<	Label::ab, RI_Tools::T_filter_func<Tdata> > filter_funcs;
	CS_Matrix<TA,TC,Tdata_real> csm;

public:		// private:
	TC period;
	MPI_Comm mpi_comm;
	
	std::unordered_map<Label::ab, std::map<TA, std::map<TAC, Tensor<Tdata>>>> Ds_ab;

public:		// private:
	using T_cal_func = std::function<void(
		const Label::ab_ab &label,
		const TA &Aa01, const TAC &Aa2, const TAC &Ab01, const TAC &Ab2,
		const Tensor<Tdata> &D_a, const Tensor<Tdata> &D_b, const Tensor<Tdata> &D_b_transpose,
		std::unordered_map<Label::ab_ab, Tensor<Tdata>> &Ds_b01,
		std::unordered_map<Label::ab_ab, Tdata_real> &Ds_b01_csm,
		LRI_Cal_Tools<TA,TC,Tdata> &tools)>;
	std::unordered_map<Label::ab_ab, T_cal_func> cal_funcs;
	void set_cal_funcs_b01();
	void set_cal_funcs_bx2();
};

#include "LRI.hpp"
#include "LRI-set.hpp"
#include "LRI-cal.hpp"
#include "LRI-cal-b01.hpp"
#include "LRI-cal-bx2.hpp"
