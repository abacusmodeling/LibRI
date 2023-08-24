// ===================
//  Author: Peize Lin
//  date: 2022.10.21
// ===================

#pragma once

#include "Save_Load.h"

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void Save_Load<TA,Tcell,Ndim,Tdata>::save(const std::string &name, const Label::ab &label)
{
	this->saves[name].Ds_ab = std::move(this->lri.Ds_ab[label]);
	this->saves[name].index_Ds_ab = std::move(this->lri.index_Ds_ab[label]);
	this->saves[name].csm_uplimits = std::move(this->lri.csm_uplimits[label]);
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void Save_Load<TA,Tcell,Ndim,Tdata>::load(const std::string &name, const Label::ab &label)
{
	assert(this->lri.Ds_ab[label].empty());
	assert(this->lri.index_Ds_ab[label].empty());

	this->lri.Ds_ab[label] = std::move(this->saves.at(name).Ds_ab);
	this->lri.index_Ds_ab[label] = std::move(this->saves.at(name).index_Ds_ab);
	this->lri.csm_uplimits[label] = std::move(this->saves.at(name).csm_uplimits);

//	auto save_name_empty = [this]() -> bool
//	{
//		for(const Label::ab &label : Label::array_ab)
//		{
//			if(!this->saves[name].Ds_ab[label].empty())	return false;
//			if(!this->saves[name].csm_uplimits_square_tensor3[label].empty())	return false;
//			if(!this->saves[name].csm_uplimits_norm_tensor3  [label].empty())	return false;
//			if(!this->saves[name].csm_uplimits_square_tensor2[label].empty())	return false;
//		}
//		return true;
//	};
//	if(save_name_empty())
//		this->saves.erase(name);
}

/*
void save_copy(const std::string &name, const Label::ab &label)
{
	saves[name].Ds_ab[label] = lri.Ds_ab[label].copy();
	saves[name].csm_uplimits_square_tensor3[label] = lri.csm.uplimits_square_tensor3[label].copy();
	saves[name].csm_uplimits_norm_tensor3[label]   = lri.csm.uplimits_norm_tensor3[label].copy();
	saves[name].csm_uplimits_square_tensor2[label] = lri.csm.uplimits_square_tensor2[label].copy();
}

void save_copy(const std::string &name)
{
	for(const Label::ab &label : Label::array_ab)
		this->save_copy(name, label);
}
*/

}