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
	using TC = std::array<Tcell,Ndim>;
	using TAC = std::pair<TA,TC>;
	using Tdata_real = Global_Func::To_Real_t<Tdata>;

	Save_Load(LRI<TA,Tcell,Ndim,Tdata> &lri_in): lri(lri_in){}

	void save(const std::string &name, const Label::ab &label);
//	void save(const std::string &name, const std::vector<Label::ab> &labels);
//	void save(const std::string &name);
	void load(const std::string &name, const Label::ab &label);
//	void load(const std::string &name, const std::vector<Label::ab> &labels);
//	void load(const std::string &name);


public:	// private:
	struct Info
	{
		std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_ab;
		std::vector<std::set<TA>> index_Ds_ab;
		typename CS_Matrix<TA,TC,Tdata_real>::Uplimits csm_uplimits;
	};
	std::map<std::string,Info> saves;

	LRI<TA,Tcell,Ndim,Tdata> &lri;
};

}

#include "Save_Load.hpp"