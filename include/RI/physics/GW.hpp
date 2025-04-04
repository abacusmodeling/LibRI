// ===================
//  Author: Minye Zhang, almost completely copied from Exx.hpp
//  date: 2022.12.18
// ===================
#pragma once

#include "GW.h"
#include "../ri/Label_Tools.h"
#include "../ri/Label.h"
#include "./symmetry/Filter_Atom_Symmetry.h"

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void G0W0<TA,Tcell,Ndim,Tdata>::set_parallel(
	const MPI_Comm &mpi_comm_in,
	const std::map<TA,Tatom_pos> &atoms_pos_in,
	const std::array<Tatom_pos,Ndim> &latvec_in,
	const std::array<Tcell,Ndim> &period_in)
{
	this->mpi_comm = mpi_comm_in;
	this->atoms_pos = atoms_pos_in;
	this->latvec = latvec_in;
	this->period = period_in;

	this->lri.set_parallel(
		this->mpi_comm, this->atoms_pos, this->latvec, this->period,
		{Label::ab_ab::a0b0_a1b1, Label::ab_ab::a0b0_a1b2, Label::ab_ab::a0b0_a2b1, Label::ab_ab::a0b0_a2b2});
	this->flag_finish.stru = true;
	//if()
		// this->post_2D.set_parallel(this->mpi_comm, this->atoms_pos, this->period);
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void G0W0<TA,Tcell,Ndim,Tdata>::set_symmetry(
	const bool flag_symmetry,
	const std::map<std::pair<TA,TA>, std::set<TC>> &irreducible_sector)
{
	if(flag_symmetry)
		this->lri.filter_atom = std::make_shared<Filter_Atom_Symmetry<TA,TC,Tdata>>(
			this->period, irreducible_sector);
	else
		this->lri.filter_atom = std::make_shared<Filter_Atom<TA,TAC>>();
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void G0W0<TA,Tcell,Ndim,Tdata>::set_Cs(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Cs,
	const Tdata_real &threshold_C)
{
	this->lri.set_tensors_map2(
		Cs,
		{Label::ab::a, Label::ab::b},
		{{"threshold_filter", threshold_C}} );
	this->flag_finish.C = true;
}

template <typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void G0W0<TA, Tcell, Ndim, Tdata>::set_Wc(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> Wc_tau,
	const Tdata_real &threshold_W)
{
	// setup screened Coulomb interaction
	this->lri.set_tensors_map2(
		Wc_tau,
		{Label::ab::a0b0},
		{{"threshold_filter", threshold_W}} );
	this->flag_finish.W = true;
}

template <typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void G0W0<TA, Tcell, Ndim, Tdata>::cal_Sigc(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> gf_tau,
	const Tdata_real &threshold_G)
{
	assert(this->flag_finish.stru);
	assert(this->flag_finish.C);
	assert(this->flag_finish.W);
	// setup Green's function
	this->lri.set_tensors_map2(
		gf_tau,
		{Label::ab::a1b1, Label::ab::a1b2, Label::ab::a2b1, Label::ab::a2b2},
		{{"threshold_filter", threshold_G}} );

	this->Sigc_tau.clear();
	this->lri.cal_loop3(
		{Label::ab_ab::a0b0_a1b1,
		 Label::ab_ab::a0b0_a1b2,
		 Label::ab_ab::a0b0_a2b1,
		 Label::ab_ab::a0b0_a2b2},
		this->Sigc_tau);
}

template <typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void G0W0<TA, Tcell, Ndim, Tdata>::free_Wc()
{
	const std::vector<Label::ab> label_list = {Label::ab::a0b0,};
	const std::string save_name = Label_Tools::get_name(label_list);
	for(const Label::ab &label : label_list)
		this->lri.data_ab_name.erase(label);
	this->lri.data_pool.erase(save_name);
	this->flag_finish.W = false;
}

template <typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void G0W0<TA, Tcell, Ndim, Tdata>::free_G()
{
	const std::vector<Label::ab> label_list = {Label::ab::a1b1, Label::ab::a1b2, Label::ab::a2b1, Label::ab::a2b2};
	const std::string save_name = Label_Tools::get_name(label_list);
	for(const Label::ab &label : label_list)
		this->lri.data_ab_name.erase(label);
	this->lri.data_pool.erase(save_name);
}

} // namespace RI