// ===================
//  Author: Peize Lin
//  date: 2022.10.21
// ===================

#pragma once

#include "RI/ri/Label.h"
#include <string>
#include <stdexcept>

namespace Lable_Test
{
	static std::string get_name(const RI::Label::ab &label)
	{
		switch(label)
		{
			case RI::Label::ab::a:		return "a";
			case RI::Label::ab::b:		return "b";
			case RI::Label::ab::a0b0:	return "a0b0";
			case RI::Label::ab::a0b1:	return "a0b1";
			case RI::Label::ab::a0b2:	return "a0b2";
			case RI::Label::ab::a1b0:	return "a1b0";
			case RI::Label::ab::a1b1:	return "a1b1";
			case RI::Label::ab::a1b2:	return "a1b2";
			case RI::Label::ab::a2b0:	return "a2b0";
			case RI::Label::ab::a2b1:	return "a2b1";
			case RI::Label::ab::a2b2:	return "a2b2";
			default:	throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
		}
	}

	static std::string get_name(const RI::Label::ab_ab &label)
	{
		switch(label)
		{
			case RI::Label::ab_ab::a1b1_a2b2:		return "a1b1_a2b2";
			case RI::Label::ab_ab::a1b0_a2b2:		return "a1b0_a2b2";
			case RI::Label::ab_ab::a1b0_a2b1:		return "a1b0_a2b1";
			case RI::Label::ab_ab::a0b1_a2b2:		return "a0b1_a2b2";
			case RI::Label::ab_ab::a0b0_a2b2:		return "a0b0_a2b2";
			case RI::Label::ab_ab::a0b0_a2b1:		return "a0b0_a2b1";
			case RI::Label::ab_ab::a0b1_a1b2:		return "a0b1_a1b2";
			case RI::Label::ab_ab::a0b0_a1b2:		return "a0b0_a1b2";
			case RI::Label::ab_ab::a0b0_a1b1:		return "a0b0_a1b1";
			case RI::Label::ab_ab::a1b2_a2b1:		return "a1b2_a2b1";
			case RI::Label::ab_ab::a1b2_a2b0:		return "a1b2_a2b0";
			case RI::Label::ab_ab::a1b1_a2b0:		return "a1b1_a2b0";
			case RI::Label::ab_ab::a0b2_a2b1:		return "a0b2_a2b1";
			case RI::Label::ab_ab::a0b2_a2b0:		return "a0b2_a2b0";
			case RI::Label::ab_ab::a0b1_a2b0:		return "a0b1_a2b0";
			case RI::Label::ab_ab::a0b2_a1b1:		return "a0b2_a1b1";
			case RI::Label::ab_ab::a0b2_a1b0:		return "a0b2_a1b0";
			case RI::Label::ab_ab::a0b1_a1b0:		return "a0b1_a1b0";
			default:	throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
		}
	}
}