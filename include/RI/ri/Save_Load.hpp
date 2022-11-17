// ===================
//  Author: Peize Lin
//  date: 2022.10.21
// ===================

#pragma once

#include "Save_Load.h"

namespace RI
{

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Save_Load<TA,Tcell,Ndim,Tdata>::save(const std::string &name, const Label::ab &label)
{
	this->saves[name].Ds_ab.reserve(Label::array_ab.size());
	this->saves[name].csm_uplimits_square_tensor3.reserve(Label::array_ab.size());
	this->saves[name].csm_uplimits_norm_tensor3  .reserve(Label::array_ab.size());
	this->saves[name].csm_uplimits_square_tensor2.reserve(Label::array_ab.size());

	this->saves[name].Ds_ab[label] = std::move(this->lri.Ds_ab[label]);
	this->saves[name].csm_uplimits_square_tensor3[label] = std::move(this->lri.csm.uplimits_square_tensor3[label]);
	this->saves[name].csm_uplimits_norm_tensor3  [label] = std::move(this->lri.csm.uplimits_norm_tensor3  [label]);
	this->saves[name].csm_uplimits_square_tensor2[label] = std::move(this->lri.csm.uplimits_square_tensor2[label]);
}

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Save_Load<TA,Tcell,Ndim,Tdata>::save(const std::string &name)
{
	for(const Label::ab &label : Label::array_ab)
		this->save(name, label);
}

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Save_Load<TA,Tcell,Ndim,Tdata>::save(const std::string &name, const std::vector<Label::ab> &labels)
{
	for(const Label::ab &label : labels)
		this->save(name, label);
}

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Save_Load<TA,Tcell,Ndim,Tdata>::load(const std::string &name, const Label::ab &label)
{
	assert(this->lri.Ds_ab[label].empty());
	for(std::size_t i=0; i<this->lri.csm.uplimits_square_tensor3[label].size(); ++i)
		assert(this->lri.csm.uplimits_square_tensor3[label][i].empty());
	for(std::size_t i=0; i<this->lri.csm.uplimits_norm_tensor3[label].size(); ++i)
		assert(this->lri.csm.uplimits_norm_tensor3[label][i].empty());
	assert(this->lri.csm.uplimits_square_tensor2[label].empty());

	this->lri.Ds_ab[label] = std::move(this->saves.at(name).Ds_ab[label]);
	this->lri.csm.uplimits_square_tensor3[label] = std::move(this->saves.at(name).csm_uplimits_square_tensor3[label]);
	this->lri.csm.uplimits_norm_tensor3  [label] = std::move(this->saves.at(name).csm_uplimits_norm_tensor3  [label]);
	this->lri.csm.uplimits_square_tensor2[label] = std::move(this->saves.at(name).csm_uplimits_square_tensor2[label]);

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

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Save_Load<TA,Tcell,Ndim,Tdata>::load(const std::string &name)
{
	for(const Label::ab &label : Label::array_ab)
		this->load(name, label);
	this->saves.erase(name);
}

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Save_Load<TA,Tcell,Ndim,Tdata>::load(const std::string &name, const std::vector<Label::ab> &labels)
{
	for(const Label::ab &label : labels)
		this->load(name, label);
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