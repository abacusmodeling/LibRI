// ===================
//  Author: Peize Lin
//  date: 2026.11.10
// ===================

#pragma once

#ifndef __GPU_RI

#include "Filter_Atom_Symmetry.h"

namespace RI
{

template<typename TA, typename TC, typename Tdata>
class Filter_Atom_Symmetry_Wrapper: public Filter_Atom_Symmetry<TA,TC,Tdata>
{
  public:
	Filter_Atom_Symmetry_Wrapper(
		const TC &period,
		const std::map<std::pair<TA,TA>, std::set<TC>> &irsec,
		const LRI_Cal_Mode &cal_mode_in)
		: Filter_Atom_Symmetry<TA,TC,Tdata>(period, irsec){}
};

}

#else

#include "../../ri/Filter_Atom.h"
#include "Filter_Atom_Symmetry.h"
#include "Filter_Atom_Symmetry_GPU.h"
#include "../../ri/Label_Tools.h"

#include <stdexcept>

namespace RI
{

template<typename TA, typename TC, typename Tdata>
class Filter_Atom_Symmetry_Wrapper: public Filter_Atom<TA, std::pair<TA, TC>>
{
  public:
	using TAC = std::pair<TA, TC>;

	Filter_Atom_Symmetry_Wrapper(
		const TC &period,
		const std::map<std::pair<TA,TA>, std::set<TC>> &irsec,
		const LRI_Cal_Mode &cal_mode_in)
		:cal_mode(cal_mode_in),
		 filter_CPU(period, irsec),
		 filter_GPU(period, irsec){}

	virtual bool filter_for1(const Label::ab_ab &label, const TA &A1) const override
	{
		return this->cal_mode==LRI_Cal_Mode::GPU
			? this->filter_GPU.filter_for1(label, A1)
			: this->filter_CPU.filter_for1(label, A1);
	}
	virtual bool filter_for1(const Label::ab_ab &label, const TAC &A1) const override
	{
		return this->cal_mode==LRI_Cal_Mode::GPU
			? this->filter_GPU.filter_for1(label, A1)
			: this->filter_CPU.filter_for1(label, A1);
	}

	virtual bool filter_for2(const Label::ab_ab &label, const TA &A1, const TAC &A2) const override
	{
		return this->cal_mode==LRI_Cal_Mode::GPU
			? this->filter_GPU.filter_for2(label, A1, A2)
			: this->filter_CPU.filter_for2(label, A1, A2);
	}

	virtual bool filter_for32(const Label::ab_ab &label, const TA &A1, const TAC &A2, const TAC &A3) const override
	{
		return this->cal_mode==LRI_Cal_Mode::GPU
			? this->filter_GPU.filter_for32(label, A1, A2, A3)
			: this->filter_CPU.filter_for32(label, A1, A2, A3);
	}
	virtual bool filter_for32(const Label::ab_ab &label, const TAC &A1, const TA &A2, const TAC &A3) const override
	{
		return this->cal_mode==LRI_Cal_Mode::GPU
			? false
			: this->filter_CPU.filter_for32(label, A1, A2, A3);
	}
	virtual bool filter_for32(const Label::ab_ab &label, const TAC &A1, const TAC &A2, const TAC &A3) const override
	{
		return this->cal_mode==LRI_Cal_Mode::GPU
			? false
			: this->filter_CPU.filter_for32(label, A1, A2, A3);
	}

  public:	// private
	const LRI_Cal_Mode &cal_mode;
	Filter_Atom_Symmetry<TA,TC,Tdata> filter_CPU;
	Filter_Atom_Symmetry_GPU<TA,TC,Tdata> filter_GPU;
};

}

#endif