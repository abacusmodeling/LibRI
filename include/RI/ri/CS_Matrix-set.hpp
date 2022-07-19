// ===================
//  Author: Peize Lin
//  date: 2022.06.06
// ===================

#pragma once

#include "CS_Matrix.h"
#include "CS_Matrix_Tools.h"

template<typename TA, typename Tperiod, size_t Ndim_period, typename Tdata>
void CS_Matrix<TA,Tperiod,Ndim_period,Tdata>::set_threshold(const Tdata &threshold_in)
{
	for(const Label::ab_ab &label : Label::array_ab_ab)
		threshold[label] = threshold_in;
	threshold_max = threshold_in;
}
template<typename TA, typename Tperiod, size_t Ndim_period, typename Tdata>
void CS_Matrix<TA,Tperiod,Ndim_period,Tdata>::set_threshold(const Label::ab_ab &label, const Tdata &threshold_in)
{
	threshold[label] = threshold_in;
	threshold_max = 0;
	for(const auto thr : threshold)
		threshold_max = std::max(thr, threshold_max);

}


template<typename TA, typename Tperiod, size_t Ndim_period, typename Tdata> template<typename T>
void CS_Matrix<TA,Tperiod,Ndim_period,Tdata>::set_tensor(
	const Label::ab &label,
	const T &Ds)
{
	switch(label)
	{
		case Label::ab::a:	case Label::ab::b:
			uplimits_square_tensor3[label][0] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::square_three_0, Ds);
			uplimits_square_tensor3[label][1] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::square_three_1, Ds);
			uplimits_square_tensor3[label][2] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::square_three_2, Ds);
			uplimits_norm_tensor3[label][0] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::norm_three_0, Ds);
			uplimits_norm_tensor3[label][1] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::norm_three_1, Ds);
			uplimits_norm_tensor3[label][2] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::norm_three_2, Ds);
			break;
		case Label::ab::a0b0:	case Label::ab::a0b1:	case Label::ab::a0b2:
		case Label::ab::a1b0:	case Label::ab::a1b1:	case Label::ab::a1b2:
		case Label::ab::a2b0:	case Label::ab::a2b1:	case Label::ab::a2b2:
			uplimits_square_tensor2[label] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::square_two, Ds);
			//uplimits_norm_tensor2[label] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::norm_two, Ds);
			break;
		default:
			throw std::invalid_argument("CS_Matrix::set_tensor");
	}
}