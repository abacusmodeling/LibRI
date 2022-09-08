// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include <array>
#include <stdexcept>

namespace Label
{
	enum class ab
	{
		a, b,
		a0b0, a0b1, a0b2,
		a1b0, a1b1, a1b2,
		a2b0, a2b1, a2b2
	};
	constexpr std::array<Label::ab,11> array_ab = {
		ab::a, ab::b, 
		ab::a0b0, ab::a0b1, ab::a0b2,
		ab::a1b0, ab::a1b1, ab::a1b2,
		ab::a2b0, ab::a2b1, ab::a2b2};

	enum class ab_ab
	{
		a1b1_a2b2, a1b2_a2b1,	// a0,b0
		a1b0_a2b2, a1b2_a2b0,	// a0,b1
		a1b0_a2b1, a1b1_a2b0,	// a0,b2
		a0b1_a2b2, a0b2_a2b1,	// a1,b0
		a0b0_a2b2, a0b2_a2b0,	// a1,b1
		a0b0_a2b1, a0b1_a2b0,	// a1,b2
		a0b1_a1b2, a0b2_a1b1,	// a2,b0
		a0b0_a1b2, a0b2_a1b0,	// a2,b1
		a0b0_a1b1, a0b1_a1b0	// a2,b2
	};
	constexpr std::array<Label::ab_ab,18> array_ab_ab = {
		ab_ab::a1b1_a2b2, ab_ab::a1b2_a2b1,
		ab_ab::a1b0_a2b2, ab_ab::a1b2_a2b0,
		ab_ab::a1b0_a2b1, ab_ab::a1b1_a2b0,
		ab_ab::a0b1_a2b2, ab_ab::a0b2_a2b1,
		ab_ab::a0b0_a2b2, ab_ab::a0b2_a2b0,
		ab_ab::a0b0_a2b1, ab_ab::a0b1_a2b0,
		ab_ab::a0b1_a1b2, ab_ab::a0b2_a1b1,
		ab_ab::a0b0_a1b2, ab_ab::a0b2_a1b0,
		ab_ab::a0b0_a1b1, ab_ab::a0b1_a1b0
	};

	inline int get_unused_a(const ab_ab &label)
	{
		switch(label)
		{
			case ab_ab::a1b1_a2b2:	case ab_ab::a1b2_a2b1:
			case ab_ab::a1b0_a2b2:	case ab_ab::a1b2_a2b0:
			case ab_ab::a1b0_a2b1:	case ab_ab::a1b1_a2b0:
				return 0;
			case ab_ab::a0b1_a2b2:	case ab_ab::a0b2_a2b1:
			case ab_ab::a0b0_a2b2:	case ab_ab::a0b2_a2b0:
			case ab_ab::a0b0_a2b1:	case ab_ab::a0b1_a2b0:
				return 1;
			case ab_ab::a0b1_a1b2:	case ab_ab::a0b2_a1b1:
			case ab_ab::a0b0_a1b2:	case ab_ab::a0b2_a1b0:
			case ab_ab::a0b0_a1b1:	case ab_ab::a0b1_a1b0:
				return 2;
			default:
				throw std::invalid_argument("Label::get_unused_a");
		}
	}

	inline int get_unused_b(const ab_ab &label)
	{
		switch(label)
		{
			case ab_ab::a1b1_a2b2:	case ab_ab::a1b2_a2b1:
			case ab_ab::a0b1_a2b2:	case ab_ab::a0b2_a2b1:
			case ab_ab::a0b1_a1b2:	case ab_ab::a0b2_a1b1:
				return 0;
			case ab_ab::a1b0_a2b2:	case ab_ab::a1b2_a2b0:
			case ab_ab::a0b0_a2b2:	case ab_ab::a0b2_a2b0:
			case ab_ab::a0b0_a1b2:	case ab_ab::a0b2_a1b0:
				return 1;
			case ab_ab::a1b0_a2b1:	case ab_ab::a1b1_a2b0:
			case ab_ab::a0b0_a2b1:	case ab_ab::a0b1_a2b0:
			case ab_ab::a0b0_a1b1:	case ab_ab::a0b1_a1b0:
				return 2;
			default:
				throw std::invalid_argument("Label::get_unused_b");
		}
	}

	inline int get_a(const ab &label)
	{
		switch(label)
		{
			case ab::a0b0:	case ab::a0b1:	case ab::a0b2:	return 0;
			case ab::a1b0:	case ab::a1b1:	case ab::a1b2:	return 1;
			case ab::a2b0:	case ab::a2b1:	case ab::a2b2:	return 2;
			default:	throw std::invalid_argument("Label::get_a");
		}
	}

	inline int get_b(const ab &label)
	{
		switch(label)
		{
			case ab::a0b0:	case ab::a1b0:	case ab::a2b0:	return 0;
			case ab::a0b1:	case ab::a1b1:	case ab::a2b1:	return 1;
			case ab::a0b2:	case ab::a1b2:	case ab::a2b2:	return 2;
			default:	throw std::invalid_argument("Label::get_b");
		}
	}
}