// ===================
//  Author: Peize Lin
//  date: 2022.08.12
// ===================

#pragma once

#include "RI/global/Tensor.h"

#include <memory.h>
#include <cassert>
#include <stdexcept>

namespace LRI_Cal_Aux
{
	template<typename Tdata>
	inline Tensor<Tdata> tensor3_merge(const Tensor<Tdata> &D, const bool flag_01_2)
	{
		assert(D.shape.size()==3);
		if(flag_01_2)
			return Tensor<Tdata>( {D.shape[0]*D.shape[1], D.shape[2]}, D.data );
		else
			return Tensor<Tdata>( {D.shape[0], D.shape[1]*D.shape[2]}, D.data );
	};

	template<typename Tdata>
	Tensor<Tdata> tensor3_transpose(const Tensor<Tdata> &D)
	{
		assert(D.shape.size()==3);
		Tensor<Tdata> D_new({D.shape[1], D.shape[0], D.shape[2]});
		for(size_t i0=0; i0<D.shape[0]; ++i0)
			for(size_t i1=0; i1<D.shape[1]; ++i1)
			{
				memcpy(
					D_new.ptr()+(i1*D.shape[0]+i0)*D.shape[2],
					D.ptr()+(i0*D.shape[1]+i1)*D.shape[2],
					D.shape[2]*sizeof(Tdata));
			}
		return D_new;
	};

	template<typename Tdata>
	inline void add_D(const Tensor<Tdata> &D_add, Tensor<Tdata> &D_result)
	{
		if(D_result.empty())
			D_result = D_add;
		else
			D_result = D_result + D_add;
	};

	inline int judge_x(const Label::ab_ab &label)
	{
		switch(label)
		{
			case Label::ab_ab::a1b0_a2b2:	case Label::ab_ab::a1b2_a2b0:
			case Label::ab_ab::a0b0_a2b2:	case Label::ab_ab::a0b2_a2b0:
			case Label::ab_ab::a0b0_a1b2:	case Label::ab_ab::a0b2_a1b0:
				return 0;
			case Label::ab_ab::a1b1_a2b2:	case Label::ab_ab::a1b2_a2b1:
			case Label::ab_ab::a0b1_a2b2:	case Label::ab_ab::a0b2_a2b1:
			case Label::ab_ab::a0b1_a1b2:	case Label::ab_ab::a0b2_a1b1:
				return 1;
			default:
				return -1;
		}
	};

	inline Label::ab get_abx(const Label::ab_ab &label)
	{
		switch(label)
		{
			case Label::ab_ab::a0b0_a1b2:	case Label::ab_ab::a0b0_a2b2:	return Label::ab::a0b0;
			case Label::ab_ab::a0b1_a1b2:	case Label::ab_ab::a0b1_a2b2:	return Label::ab::a0b1;
			case Label::ab_ab::a0b2_a1b0:	case Label::ab_ab::a1b0_a2b2:	return Label::ab::a1b0;
			case Label::ab_ab::a0b2_a1b1:	case Label::ab_ab::a1b1_a2b2:	return Label::ab::a1b1;
			case Label::ab_ab::a0b2_a2b0:	case Label::ab_ab::a1b2_a2b0:	return Label::ab::a2b0;
			case Label::ab_ab::a0b2_a2b1:	case Label::ab_ab::a1b2_a2b1:	return Label::ab::a2b1;
			default:	throw std::invalid_argument("get_abx");
		}
	};	
}