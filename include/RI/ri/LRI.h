// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "Label.h"
#include "global/Tensor.h"
#include "CS_Matrix.h"
#include "RI_Tools.h"
#include "global/Global_Func-2.h"

#include <map>
#include <set>
#include <array>
#include <unordered_map>
#include <functional>

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
class LRI
{
public:
	using TAC = std::pair<TA,std::array<Tcell,Ndim>>;

	//std::map<TA, std::map<TAC, Tensor<Tdata>> Ds_a;
	//std::map<TA, std::map<TAC, Tensor<Tdata>> Ds_b;
	std::unordered_map<Label::ab, std::map<TA, std::map<TAC, Tensor<Tdata>>>> Ds_ab;
	std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_result;

	CS_Matrix< TA,TA,Ndim,
		Global_Func::To_Real_t<Tdata> > csm;

	LRI();
	
	void set_tensor2(
		const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_local,
		const Label::ab &label,
		const Global_Func::To_Real_t<Tdata> &threshold);
	//void set_tensor3(
	//	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_local,
	//	const std::string &label,
	//	const Tdata &threshold);

	void cal(const std::vector<Label::ab_ab> &lables);

	std::unordered_map<	Label::ab, RI_Tools::T_filter_func<Tdata> > filter_funcs;

	std::function<std::set<TA>()> list_Aa01;
	std::function<std::set<TAC>()> list_Aa2;
	std::function<std::set<TAC>()> list_Ab01;
	std::function<std::set<TAC>()> list_Ab2;

	std::array<Tcell,Ndim> period;
};

#include "LRI.hpp"
#include "LRI-set.hpp"
#include "LRI-cal.hpp"