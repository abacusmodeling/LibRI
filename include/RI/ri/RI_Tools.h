// ===================
//  Author: Peize Lin
//  date: 2022.07.01
// ===================

#pragma once

#include "global/Tensor.h"
#include "global/Global_Func-2.h"
#include <map>

namespace RI_Tools
{
	template<typename Tdata>
	using T_filter_func =
		std::function<bool(
			const Tensor<Tdata> &D,
			const Global_Func::To_Real_t<Tdata> &threshold)>;

	template<typename TA, typename TAp, typename Tdata>
	std::map<TA, std::map<TAp, Tensor<Tdata>>> filter(
		const std::map<TA, std::map<TAp, Tensor<Tdata>>> &Ds,
		const T_filter_func<Tdata> &filter_func,
		const Global_Func::To_Real_t<Tdata> &threshold);

	template<typename TA, typename TP, typename Tdata>
	std::map<TA, std::map<std::pair<TA,TP>, Tensor<Tdata>>> cal_period(
		const std::map<TA, std::map<std::pair<TA,TP>, Tensor<Tdata>>> &Ds,
		const TP &period);		
}

#include "RI_Tools.hpp"