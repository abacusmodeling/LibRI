// ===================
//  Author: Peize Lin
//  date: 2024.08.17
// ===================

#pragma once

#include "Label.h"

namespace RI
{

template<typename TA, typename TAC>
class Filter_Atom
{
public:
	virtual bool filter_for1(const Label::ab_ab &label, const TA  &A1) const { return false; }
		                     //    a01b01_a2b01:                   Aa01
		                     //    a01b01_a2b2:                    Aa01
		                     //    a01b2_a2b01:                    Aa01
	virtual bool filter_for1(const Label::ab_ab &label, const TAC &A1) const { return false; }
		                     //    a01b01_a01b01:                  Aa2
		                     //    a01b01_a01b2:                   Ab01



	virtual bool filter_for2(const Label::ab_ab &label, const TA  &A1, const TAC &A2) const { return false; }
		                     //    a01b01_a2b01:                   Aa01,          Ab01
		                     //    a01b01_a2b2:                    Aa01,          Ab2
		                     //    a01b2_a2b01:                    Aa01,          Ab01
	virtual bool filter_for2(const Label::ab_ab &label, const TAC &A1, const TA  &A2) const { return false; }
		                     //    a01b01_a01b2:                   Ab01,          Aa01
	virtual bool filter_for2(const Label::ab_ab &label, const TAC &A1, const TAC &A2) const { return false; }
		                     //    a01b01_a01b01:                  Aa2,           Ab01



	virtual bool filter_for31(const Label::ab_ab &label, const TA  &A1, const TAC &A2, const TAC &A3) const { return false; }
		                     //    a01b01_a2b01:                    Aa01,          Ab01,          Aa2
		                     //    a01b01_a2b2:                     Aa01,          Ab2,           Aa2
		                     //    a01b2_a2b01:                     Aa01,          Ab01,          Ab2
	virtual bool filter_for31(const Label::ab_ab &label, const TAC &A1, const TA  &A2, const TAC &A3) const { return false; }
		                     //    a01b01_a01b2:                    Ab01,          Aa01,          Ab2
	virtual bool filter_for31(const Label::ab_ab &label, const TAC &A1, const TAC &A2, const TA  &A3) const { return false; }
		                     //    a01b01_a01b01:                   Aa2,           Ab01,          Aa01



	virtual bool filter_for32(const Label::ab_ab &label, const TA  &A1, const TAC &A2, const TAC &A3) const { return false; }
		                     //    a01b01_a2b01:                    Aa01,          Ab01,          Ab2
		                     //    a01b01_a2b2:                     Aa01,          Ab2,           Ab01
		                     //    a01b2_a2b01:                     Aa01,          Ab01,          Aa2
	virtual bool filter_for32(const Label::ab_ab &label, const TAC &A1, const TA  &A2, const TAC &A3) const { return false; }
		                     //    a01b01_a01b2:                    Ab01,          Aa01,          Aa2
	virtual bool filter_for32(const Label::ab_ab &label, const TAC &A1, const TAC &A2, const TAC &A3) const { return false; }
		                     //    a01b01_a01b01:                   Aa2,           Ab01,          Ab2



  #ifdef __GPU_RI
	virtual bool filter_GPU_for11(const Label::ab_ab &label, const TA  &A1) const { return false; }
	virtual bool filter_GPU_for11(const Label::ab_ab &label, const TAC &A1) const { return false; }

	virtual bool filter_GPU_for12(const Label::ab_ab &label, const TA  &A1) const { return false; }
	virtual bool filter_GPU_for12(const Label::ab_ab &label, const TAC &A1) const { return false; }

	virtual bool filter_GPU_for13(const Label::ab_ab &label, const TA  &A1) const { return false; }
	virtual bool filter_GPU_for13(const Label::ab_ab &label, const TAC &A1) const { return false; }

	virtual bool filter_GPU_for21(const Label::ab_ab &label, const TA  &A1, const TAC &A2) const { return false; }
	virtual bool filter_GPU_for21(const Label::ab_ab &label, const TAC &A1, const TA  &A2) const { return false; }
	virtual bool filter_GPU_for21(const Label::ab_ab &label, const TAC &A1, const TAC &A2) const { return false; }

	virtual bool filter_GPU_for22(const Label::ab_ab &label, const TA  &A1, const TAC &A2) const { return false; }
	virtual bool filter_GPU_for22(const Label::ab_ab &label, const TAC &A1, const TA  &A2) const { return false; }
	virtual bool filter_GPU_for22(const Label::ab_ab &label, const TAC &A1, const TAC &A2) const { return false; }

	virtual bool filter_GPU_for23(const Label::ab_ab &label, const TA  &A1, const TAC &A2) const { return false; }
	virtual bool filter_GPU_for23(const Label::ab_ab &label, const TAC &A1, const TA  &A2) const { return false; }
	virtual bool filter_GPU_for23(const Label::ab_ab &label, const TAC &A1, const TAC &A2) const { return false; }

	virtual bool filter_GPU_for31(const Label::ab_ab &label, const TA  &A1, const TAC &A2, const TAC &A3) const { return false; }
	virtual bool filter_GPU_for31(const Label::ab_ab &label, const TAC &A1, const TA  &A2, const TAC &A3) const { return false; }
	virtual bool filter_GPU_for31(const Label::ab_ab &label, const TAC &A1, const TAC &A2, const TA  &A3) const { return false; }
	virtual bool filter_GPU_for31(const Label::ab_ab &label, const TAC &A1, const TAC &A2, const TAC &A3) const { return false; }

	virtual bool filter_GPU_for32(const Label::ab_ab &label, const TA  &A1, const TAC &A2, const TAC &A3) const { return false; }
	virtual bool filter_GPU_for32(const Label::ab_ab &label, const TAC &A1, const TA  &A2, const TAC &A3) const { return false; }
	virtual bool filter_GPU_for32(const Label::ab_ab &label, const TAC &A1, const TAC &A2, const TA  &A3) const { return false; }
	virtual bool filter_GPU_for32(const Label::ab_ab &label, const TAC &A1, const TAC &A2, const TAC &A3) const { return false; }
  #endif



	virtual ~Filter_Atom()=default;
};

}