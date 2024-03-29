// ===================
//  Author: Peize Lin
//  date: 2022.08.12
// ===================

#pragma once

#include "../global/Tensor.h"
#include "Label.h"
#include "../global/Map_Operator.h"

#include <map>
#include <memory.h>
#include <cassert>
#include <stdexcept>

namespace RI
{

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
	}

	template<typename Tdata>
	Tensor<Tdata> tensor3_transpose(const Tensor<Tdata> &D)
	{
		assert(D.shape.size()==3);
		Tensor<Tdata> D_new({D.shape[1], D.shape[0], D.shape[2]});
		for(std::size_t i0=0; i0<D.shape[0]; ++i0)
			for(std::size_t i1=0; i1<D.shape[1]; ++i1)
			{
				memcpy(
					D_new.ptr()+(i1*D.shape[0]+i0)*D.shape[2],
					D.ptr()+(i0*D.shape[1]+i1)*D.shape[2],
					D.shape[2]*sizeof(Tdata));
			}
		return D_new;
	}

	template<typename Tdata>
	inline void add_D(const Tensor<Tdata> &D_add, Tensor<Tdata> &D_result)
	{
		if(D_result.empty())
			D_result = D_add;
		else
			D_result = D_result + D_add;
	}

	/*
	template<typename T>
	inline void add_Ds(const std::vector<T> &Ds_add, std::vector<T> &Ds_result)
	{
		assert(Ds_add.size()==Ds_result.size());
		using namespace Map_Operator;						// tmp
		for(std::size_t i=0; i<Ds_result.size(); ++i)
			Ds_result[i] = Ds_result[i] + Ds_add[i];		// tmp
	}
	*/

	template<typename TA, typename TAC, typename Tdata>
	void add_Ds(
		std::vector<std::map<TA, std::map<TAC, Tensor<Tdata>>>> &&Ds_add,
		std::vector<std::map<TA, std::map<TAC, Tensor<Tdata>>>> &Ds_result)
	{
		assert(Ds_add.size()==Ds_result.size());
		for(std::size_t i=0; i<Ds_result.size(); ++i)
			for(auto &&Ds_add_A : Ds_add[i])
				for(auto &&Ds_add_B : Ds_add_A.second)
				{
					Tensor<Tdata> &D_result = Ds_result[i][Ds_add_A.first][Ds_add_B.first];
					if(D_result.empty())
						D_result = std::move(Ds_add_B.second);
					else
						D_result = D_result + Ds_add_B.second;
				}
		Ds_add.clear();
		Ds_add.resize(Ds_result.size());
	}

	template<typename T>
	inline bool judge_Ds_empty(const std::vector<T> &Ds)
	{
		for(const T &D : Ds)
			if(!D.empty())
				return false;
		return true;
	}

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
	}

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
	}

	template<typename Tkey0, typename Tkey1, typename Tdata>
	std::map<Tkey0, std::map<Tkey1, Tensor<Tdata>>>
	cal_Ds_transpose(const std::map<Tkey0, std::map<Tkey1, Tensor<Tdata>>> &Ds)
	{
		std::map<Tkey0, std::map<Tkey1, Tensor<Tdata>>> Ds_transpose;
		for(const auto &Ds0 : Ds)
			for(const auto &Ds1 : Ds0.second)
				Ds_transpose[Ds0.first][Ds1.first] = tensor3_transpose(Ds1.second);
		return Ds_transpose;
	}
}

}