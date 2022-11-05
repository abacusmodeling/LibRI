// ===================
//  Author: Peize Lin
//  date: 2022.10.21
// ===================

#pragma once

#include "../global/Tensor.h"
#include "Label.h"
#include "../global/Global_Func-1.h"

#include <array>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>

namespace RI
{

	template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
	class LRI;

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
class Save_Load
{
public:
	using TAC = std::pair<TA,std::array<Tcell,Ndim>>;
	using Tdata_real = Global_Func::To_Real_t<Tdata>;

	Save_Load(LRI<TA,Tcell,Ndim,Tdata> &lri_in): lri(lri_in){}

	void save(const std::string &name, const Label::ab &label);
	void save(const std::string &name, const std::vector<Label::ab> &labels);
	void save(const std::string &name);
	void load(const std::string &name, const Label::ab &label);
	void load(const std::string &name, const std::vector<Label::ab> &labels);
	void load(const std::string &name);

public:	// private:
	struct Info
	{
		std::unordered_map<Label::ab, std::map<TA, std::map<TAC, Tensor<Tdata>>>> Ds_ab;
		std::unordered_map<Label::ab, std::array< std::map<TA,std::map<TAC,Tdata_real>> ,3>> csm_uplimits_square_tensor3;
		std::unordered_map<Label::ab, std::array< std::map<TA,std::map<TAC,Tdata_real>> ,3>> csm_uplimits_norm_tensor3;
		std::unordered_map<Label::ab, std::map<TA,std::map<TAC,Tdata_real>>> csm_uplimits_square_tensor2;
		//std::unordered_map<Label::ab, std::map<TA,std::map<TAC,Tdata_real>>> csm_uplimits_norm_tensor2;
	};
	std::map<std::string,Info> saves;

	LRI<TA,Tcell,Ndim,Tdata> &lri;
};

}

#include "Save_Load.hpp"