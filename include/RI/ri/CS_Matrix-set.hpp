// ===================
//  Author: Peize Lin
//  date: 2022.06.06
// ===================

#pragma once

#include "CS_Matrix.h"
#include "CS_Matrix_Tools.h"

namespace RI
{

template<typename TA, typename TC, typename Tdata>
CS_Matrix<TA,TC,Tdata>::CS_Matrix()
{
	this->set_threshold(0);
}

template<typename TA, typename TC, typename Tdata>
void CS_Matrix<TA,TC,Tdata>::set_threshold(const Tdata &threshold_in)
{
	for(const Label::ab_ab &label : Label::array_ab_ab)
		this->threshold[label] = threshold_in;
}
template<typename TA, typename TC, typename Tdata>
void CS_Matrix<TA,TC,Tdata>::set_threshold(const Label::ab_ab &label, const Tdata &threshold_in)
{
	this->threshold[label] = threshold_in;
}


template<typename TA, typename TC, typename Tdata> template<typename T>
void CS_Matrix<TA,TC,Tdata>::set_tensor(
	const Label::ab &label,
	const T &Ds)
{
	switch(label)
	{
		case Label::ab::a:	case Label::ab::b:
			this->uplimits_square_tensor3[label][0] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::square_three_0, Ds);
			this->uplimits_square_tensor3[label][1] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::square_three_1, Ds);
			this->uplimits_square_tensor3[label][2] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::square_three_2, Ds);
			this->uplimits_norm_tensor3[label][0] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::norm_three_0, Ds);
			this->uplimits_norm_tensor3[label][1] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::norm_three_1, Ds);
			this->uplimits_norm_tensor3[label][2] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::norm_three_2, Ds);
			break;
		case Label::ab::a0b0:	case Label::ab::a0b1:	case Label::ab::a0b2:
		case Label::ab::a1b0:	case Label::ab::a1b1:	case Label::ab::a1b2:
		case Label::ab::a2b0:	case Label::ab::a2b1:	case Label::ab::a2b2:
			this->uplimits_square_tensor2[label] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::square_two, Ds);
			//this->uplimits_norm_tensor2[label] = CS_Matrix_Tools::cal_uplimit(CS_Matrix_Tools::Uplimit_Type::norm_two, Ds);
			break;
		default:
			throw std::invalid_argument("CS_Matrix::set_tensor");
	}
}

}