// ===================
//  Author: Peize Lin
//  date: 2022.06.06
// ===================

#pragma once

#include "CS_Matrix.h"
#include "CS_Matrix_Tools.h"
#include "global/Global_Func-1.h"
#include "global/Array_Operator.h"

#include <stdexcept>

template<typename TA, typename Tperiod, size_t Ndim_period, typename Tdata>
void CS_Matrix<TA,Tperiod,Ndim_period,Tdata>::set_label_A(
	const Label::ab_ab &label_in,
	const TA &Aa01, const TAp &Aa2, const TAp &Ab01, const TAp &Ab2,
	const std::array<Tperiod,Ndim_period> &period)
{
	using namespace Array_Operator;

	// 每种实现如 LRI_01_01 要写一份
	auto get_Ab = [&Ab01, &Ab2](const int xb) -> TAp
	{
		switch(xb)
		{
			case 0:	case 1:	return Ab01;
			case 2:			return Ab2;
			default:		throw std::invalid_argument("get_Ab");
		}
	};
	auto get_Aa_Ab = [&Aa01, &Aa2, &period, &get_Ab](const int xa, const int xb) -> std::pair<TA,TAp>
	{
		const TAp Ab = get_Ab(xb);
		switch(xa)
		{
			case 0:	case 1:	return {Aa01, Ab};
			case 2:			return {Aa2.first, {Ab.first, (Ab.second-Aa2.second)%period}};
			default:		throw std::invalid_argument("get_Aa_Ab");
		}
	};
	auto get_uplimit = [&get_Aa_Ab](
		const std::unordered_map<Label::ab, std::map<TA,std::map<TAp,Tdata>>> &uplimits,
		const Label::ab label_ab)
	{
		const int xa = Label::get_a(label_ab);
		const int xb = Label::get_b(label_ab);
		const std::pair<TA,TAp> &Aa_Ab = get_Aa_Ab(xa,xb);
		const TA &Aa=Aa_Ab.first;	const TAp &Ab=Aa_Ab.second;
		return Global_Func::find(uplimits.at(label_ab), Aa, Ab);
	};

	step.label = label_in;

	const int index_a = Label::get_unused_a(step.label);
	step.a_square = uplimits_square_tensor3[Label::ab::a][index_a][Aa01][Aa2];
	step.a_norm = uplimits_norm_tensor3[Label::ab::a][index_a][Aa01][Aa2];
	
	const int index_b = get_unused_b(step.label);
	step.b_square = uplimits_square_tensor3[Label::ab::b][index_b][Ab01.first][{Ab2.first, (Ab2.second-Ab01.second)%period}];
	step.b_norm = uplimits_norm_tensor3[Label::ab::b][index_b][Ab01.first][{Ab2.first, (Ab2.second-Ab01.second)%period}];
	
	const std::pair<Label::ab, Label::ab> label_split_tmp = CS_Matrix_Tools::split_label(step.label);
	step.first_square = get_uplimit(uplimits_square_tensor2, label_split_tmp.first);
	step.second_square = get_uplimit(uplimits_square_tensor2, label_split_tmp.second);
}
