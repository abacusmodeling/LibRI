// ===================
//  Author: Peize Lin
//  date: 2022.08.12
// ===================

#pragma once

#include "Label.h"
#include "RI/global/Tensor.h"
#include "RI/global/Global_Func-1.h"
#include "RI/global/Array_Operator.h"

#include <map>
#include <unordered_map>

template<typename TA, typename TC, typename Tdata>
class LRI_Cal_Tools
{
public:
	using TAC = std::pair<TA,TC>;

	LRI_Cal_Tools(
		const TC &period_in,
		std::unordered_map<Label::ab, std::map<TA, std::map<TAC, Tensor<Tdata>>>> &Ds_ab_in,
		std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_result_in)
	:period(period_in), Ds_ab(Ds_ab_in), Ds_result(Ds_result_in){}

	inline Tensor<Tdata> get_Ds_ab(const Label::ab &label,
		const TA &Aa, const TAC &Ab) const
	{
		return Global_Func::find(
			this->Ds_ab.at(label),
			Aa, Ab);
	}
	inline Tensor<Tdata> get_Ds_ab(const Label::ab &label,
		const TAC &Aa, const TAC &Ab) const
	{
		using namespace Array_Operator;
		return Global_Func::find(
			this->Ds_ab.at(label),
			Aa.first, TAC{Ab.first, (Ab.second-Aa.second)%this->period});
	}

	inline Tensor<Tdata> &get_D_result(const TA &Aa, const TAC &Ab)
	{
		return this->Ds_result[Aa][Ab];
	}

	inline Tensor<Tdata> &get_D_result(const TAC &Aa, const TAC &Ab)
	{
		using namespace Array_Operator;
		return this->Ds_result[Aa.first][TAC{Ab.first, (Ab.second-Aa.second)%period}];
	}

public:		// private:
	const TC &period;
	const std::unordered_map<Label::ab, std::map<TA, std::map<TAC, Tensor<Tdata>>>> &Ds_ab;
	std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_result;
};