// ===================
//  Author: Peize Lin
//  date: 2022.06.06
// ===================

#pragma once

#include "Label.h"

#include <array>
#include <map>
#include <unordered_map>

template<typename TA, typename Tperiod, size_t Ndim_period, typename Tdata>
class CS_Matrix
{
public:

	using TAp = std::pair<TA,std::array<Tperiod,Ndim_period>>;

	std::unordered_map<Label::ab_ab, Tdata> threshold;
	void set_threshold(const Tdata &threshold_in);
	void set_threshold(const Label::ab_ab &label, const Tdata &threshold_in);
	Tdata threshold_max = 0;

	template<typename T> void set_tensor( const Label::ab &label, const T &Ds);

	void set_label_A(
		const Label::ab_ab &label_in,
		const TA &Aa01, const TAp &Aa2, const TAp &Ab01, const TAp &Ab2,
		const std::array<Tperiod,Ndim_period> &period);

	bool filter_4(                         ) const { return step.a_square * step.first_square * step.second_square * step.b_square < this->threshold.at(step.label); }
	bool filter_3(const Tdata &uplimit_norm) const { return uplimit_norm                      * step.second_square * step.b_square < this->threshold.at(step.label); }
	bool filter_2(const Tdata &uplimit_norm) const { return uplimit_norm                                           * step.b_norm   < this->threshold.at(step.label); }
	bool filter_1(const Tdata &uplimit_norm) const { return uplimit_norm                                                           < this->threshold.at(step.label); }

private:

	struct Step
	{
		Label::ab_ab label;
		Tdata a_square;
		Tdata a_norm;
		Tdata b_square;
		Tdata b_norm;
		Tdata first_square;
		Tdata second_square;
	};
	Step step;

	std::unordered_map<Label::ab, std::array< std::map<TA,std::map<TAp,Tdata>> ,3>> uplimits_square_tensor3;
	std::unordered_map<Label::ab, std::array< std::map<TA,std::map<TAp,Tdata>> ,3>> uplimits_norm_tensor3;
	std::unordered_map<Label::ab, std::map<TA,std::map<TAp,Tdata>>> uplimits_square_tensor2;
	//std::unordered_map<Label::ab, std::map<TA,std::map<TAp,Tdata>>> uplimits_norm_tensor2;
};

#include "CS_Matrix-set.hpp"
#include "CS_Matrix-filter.hpp"