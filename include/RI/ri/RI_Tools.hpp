// ===================
//  Author: Peize Lin
//  date: 2022.07.01
// ===================

#pragma once

#include "RI_Tools.h"
#include "global/Array_Operator.h"

namespace RI_Tools
{
	template<typename TA, typename TAC, typename Tdata>
	std::map<TA, std::map<TAC, Tensor<Tdata>>> filter(
		const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds,
		const T_filter_func<Tdata> &filter_func,
		const Global_Func::To_Real_t<Tdata> &threshold)
	{
		std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_filter;
		for(const auto &Ds_tmp0 : Ds)
			for(const auto &Ds_tmp1 : Ds_tmp0.second)
				if(filter_func( Ds_tmp1.second, threshold ))
					Ds_filter[Ds_tmp0.first][Ds_tmp1.first] = Ds_tmp1.second;
		return Ds_filter;
	}


	template<typename TA, typename TP, typename Tdata>
	std::map<TA, std::map<std::pair<TA,TP>, Tensor<Tdata>>> cal_period(
		const std::map<TA, std::map<std::pair<TA,TP>, Tensor<Tdata>>> &Ds,
		const TP &period)
	{
		using namespace Array_Operator;
		std::map<TA, std::map<std::pair<TA,TP>, Tensor<Tdata>>> Ds_period;
		for(const auto &Ds_tmp0 : Ds)
		{
			for(const auto &Ds_tmp1 : Ds_tmp0.second)
			{
				Tensor<Tdata> &D_period = Ds_period[Ds_tmp0.first][{Ds_tmp1.first.first, Ds_tmp1.first.second % period}];
				if(D_period.empty())
					D_period = Ds_tmp1.second;					// share memory
				else
					D_period = D_period + Ds_tmp1.second;		// new tensor
			}
		}
		return Ds_period;
	}
}