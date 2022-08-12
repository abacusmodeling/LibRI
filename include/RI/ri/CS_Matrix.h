// ===================
//  Author: Peize Lin
//  date: 2022.06.06
// ===================

#pragma once

#include "../ri/Label.h"

#include <array>
#include <map>
#include <unordered_map>

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
class CS_Matrix
{
public:
	using TC = std::array<Tcell,Ndim>;
	using TAC = std::pair<TA,TC>;

	std::unordered_map<Label::ab_ab, Tdata> threshold;
	CS_Matrix();
	void set_threshold(const Tdata &threshold_in);
	void set_threshold(const Label::ab_ab &label, const Tdata &threshold_in);
	Tdata threshold_max = 0;

	template<typename T> void set_tensor( const Label::ab &label, const T &Ds);

	void set_label_A(
		const Label::ab_ab &label_in,
		const TA &Aa01, const TAC &Aa2, const TAC &Ab01, const TAC &Ab2,
		const TC &period);

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

	std::unordered_map<Label::ab, std::array< std::map<TA,std::map<TAC,Tdata>> ,3>> uplimits_square_tensor3;
	std::unordered_map<Label::ab, std::array< std::map<TA,std::map<TAC,Tdata>> ,3>> uplimits_norm_tensor3;
	std::unordered_map<Label::ab, std::map<TA,std::map<TAC,Tdata>>> uplimits_square_tensor2;
	//std::unordered_map<Label::ab, std::map<TA,std::map<TAC,Tdata>>> uplimits_norm_tensor2;
};

#include "CS_Matrix-set.hpp"
#include "CS_Matrix-filter.hpp"