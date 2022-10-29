// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "Exx.h"
#include "../ri/Label.h"

#include <cassert>

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::set_parallel(
	const MPI_Comm &mpi_comm,
	const std::map<TA,Tatom_pos> &atoms_pos,
	const std::array<Tatom_pos,Ndim> &latvec,
	const std::array<Tcell,Ndim> &period)
{
	this->lri.set_parallel(mpi_comm, atoms_pos, latvec, period);
	this->flag_finish.stru = true;
	//if()
		this->post_2D.set_parallel(mpi_comm, atoms_pos, period);
}

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::set_Cs(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Cs,
	const Tdata_real &threshold_C,
	const std::string &save_name_suffix)
{
	this->lri.set_tensors_map2( Cs, Label::ab::a, threshold_C );
	this->lri.set_tensors_map2( Cs, Label::ab::b, threshold_C );
	this->lri.save_load.save("Cs"+save_name_suffix, {Label::ab::a, Label::ab::b});
	this->flag_finish.C = true;
}

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::set_Vs(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Vs,
	const Tdata_real &threshold_V,
	const std::string &save_name_suffix)
{
	this->lri.set_tensors_map2( Vs, Label::ab::a0b0, threshold_V );
	this->lri.save_load.save("Vs"+save_name_suffix, Label::ab::a0b0);
	this->flag_finish.V = true;
}

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::set_Ds(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds,
	const Tdata_real &threshold_D,
	const std::string &save_name_suffix)
{
	this->lri.set_tensors_map2( Ds, Label::ab::a1b1, threshold_D );
	this->lri.set_tensors_map2( Ds, Label::ab::a1b2, threshold_D );
	this->lri.set_tensors_map2( Ds, Label::ab::a2b1, threshold_D );
	this->lri.set_tensors_map2( Ds, Label::ab::a2b2, threshold_D );
	this->lri.save_load.save("Ds"+save_name_suffix, {Label::ab::a1b1, Label::ab::a1b2, Label::ab::a2b1, Label::ab::a2b2});
	this->flag_finish.D = true;

	//if()
		this->post_2D.saves["Ds"+save_name_suffix] = this->post_2D.set_tensors_map2(Ds);
}



template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::cal_Hs(
	const std::array<std::string,3> &save_names_suffix)
{
	assert(this->flag_finish.stru);
	assert(this->flag_finish.C);
	assert(this->flag_finish.V);
	assert(this->flag_finish.D);
	
	this->lri.save_load.load("Cs"+save_names_suffix[0], {Label::ab::a, Label::ab::b});
	this->lri.save_load.load("Vs"+save_names_suffix[1], Label::ab::a0b0);
	this->lri.save_load.load("Ds"+save_names_suffix[2], {Label::ab::a1b1, Label::ab::a1b2, Label::ab::a2b1, Label::ab::a2b2});

	this->Hs.clear();
	this->lri.coefficient = nullptr;
	this->lri.cal({
		Label::ab_ab::a0b0_a1b1,
		Label::ab_ab::a0b0_a1b2,
		Label::ab_ab::a0b0_a2b1,
		Label::ab_ab::a0b0_a2b2},
		this->Hs);

	//if()
		const std::map<TA,std::map<TAC,Tensor<Tdata>>> Hs_2D = this->post_2D.set_tensors_map2(this->Hs);
		this->post_2D.energy = this->post_2D.cal_energy( this->post_2D.saves["Ds"+save_names_suffix[2]], Hs_2D );

	this->lri.save_load.save("Cs"+save_names_suffix[0], {Label::ab::a, Label::ab::b});
	this->lri.save_load.save("Vs"+save_names_suffix[1], Label::ab::a0b0);
	this->lri.save_load.save("Ds"+save_names_suffix[2], {Label::ab::a1b1, Label::ab::a1b2, Label::ab::a2b1, Label::ab::a2b2});
}
