#pragma once

#include "../../ri/Filter_Atom.h"
#include "Symmetry_Filter.h"

namespace RI
{

template<typename TA, typename TC, typename Tdata>
class Filter_Atom_Symmetry: public Filter_Atom<TA, std::pair<TA, TC>>
{
  public:
	using TAC = std::pair<TA, TC>;

	Filter_Atom_Symmetry(
		const TC& period,
		const std::map<std::pair<TA,TA>, std::set<TC>>& irsec)
		:symmetry(period, irsec){}

	virtual bool filter_for1(const Label::ab_ab& label, const TA& A1) const override
	{
		switch (label)
		{
			case Label::ab_ab::a0b0_a2b1: case Label::ab_ab::a0b0_a2b2:
				return !this->symmetry.is_I_in_irreducible_sector(A1);
			default:
				return false;
		}
	}
	virtual bool filter_for1(const Label::ab_ab& label, const TAC& A1) const override
	{
		switch (label)
		{
			case Label::ab_ab::a0b0_a1b1:
				return !this->symmetry.is_I_in_irreducible_sector(A1.first);
			case Label::ab_ab::a0b0_a1b2:
				return !this->symmetry.is_J_in_irreducible_sector(A1.first);
			default:
				return false;
		}
	}

	virtual bool filter_for32(const Label::ab_ab& label, const TA& A1, const TAC& A2, const TAC& A3) const override
	{
		switch (label)
		{
			case Label::ab_ab::a0b0_a2b1: case Label::ab_ab::a0b0_a2b2:
				return !this->symmetry.in_irreducible_sector(A1, A3);
			default:
				return false;
		}
	}
	virtual bool filter_for32(const Label::ab_ab& label, const TAC& A1, const TA& A2, const TAC& A3) const override
	{
		switch (label)
		{
			case Label::ab_ab::a0b0_a1b2:
				return !this->symmetry.in_irreducible_sector(A3, A1);
			default:
				return false;
		}
	}
	virtual bool filter_for32(const Label::ab_ab& label, const TAC& A1, const TAC& A2, const TAC& A3) const override
	{
		switch (label)
		{
			case Label::ab_ab::a0b0_a1b1:
				return !this->symmetry.in_irreducible_sector(A1, A3);
			default:
				return false;
		}
	}

  public:	// private
	Symmetry_Filter<TA,TC,Tdata> symmetry;
};

}