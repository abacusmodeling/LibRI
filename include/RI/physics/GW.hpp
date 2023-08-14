// ===================
//  Author: Minye Zhang, almost completely copied from Exx.hpp
//  date: 2022.12.18
// ===================
#pragma once

#include "GW.h"
#include "../ri/Label.h"

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

	this->lri.set_parallel(this->mpi_comm, this->atoms_pos, this->latvec, this->period);
	this->flag_finish.stru = true;
	//if()
		// this->post_2D.set_parallel(this->mpi_comm, this->atoms_pos, this->period);
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void G0W0<TA,Tcell,Ndim,Tdata>::set_Cs(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Cs,
	const Tdata_real &threshold_C)
{
	this->lri.set_tensors_map2( Cs, Label::ab::a, threshold_C );
	this->lri.set_tensors_map2( Cs, Label::ab::b, threshold_C );
	this->flag_finish.C = true;
}


template <typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void G0W0<TA, Tcell, Ndim, Tdata>::cal_Sigc(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> gf_tau,
	const Tdata_real &threshold_G,
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> Wc_tau,
	const Tdata_real &threshold_W)
{
	assert(this->flag_finish.stru);
	assert(this->flag_finish.C);
	// setup Green's function
	this->lri.set_tensors_map2( gf_tau, Label::ab::a1b1, threshold_G );
	this->lri.set_tensors_map2( gf_tau, Label::ab::a1b2, threshold_G );
	this->lri.set_tensors_map2( gf_tau, Label::ab::a2b1, threshold_G );
	this->lri.set_tensors_map2( gf_tau, Label::ab::a2b2, threshold_G );

	// setup screened Coulomb interaction
	this->lri.set_tensors_map2( Wc_tau, Label::ab::a0b0, threshold_W );

	std::vector<std::map<TA, std::map<TAC, Tensor<Tdata>>>> Sigc_vec(1);
	this->lri.coefficients = {nullptr};
	this->lri.cal_loop3(
		{Label::ab_ab::a0b0_a1b1,
		 Label::ab_ab::a0b0_a1b2,
		 Label::ab_ab::a0b0_a2b1,
		 Label::ab_ab::a0b0_a2b2},
		Sigc_vec);
	this->Sigc_tau = std::move(Sigc_vec[0]);
}

} // namespace RI