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
	this->lri.save_load.save("Cs_"+save_name_suffix, {Label::ab::a, Label::ab::b});
	this->flag_finish.C = true;
}

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::set_Vs(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Vs,
	const Tdata_real &threshold_V,
	const std::string &save_name_suffix)
{
	this->lri.set_tensors_map2( Vs, Label::ab::a0b0, threshold_V );
	this->lri.save_load.save("Vs_"+save_name_suffix, Label::ab::a0b0);
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
	this->lri.save_load.save("Ds_"+save_name_suffix, {Label::ab::a1b1, Label::ab::a1b2, Label::ab::a2b1, Label::ab::a2b2});
	this->flag_finish.D = true;

	//if()
		this->post_2D.saves["Ds_"+save_name_suffix] = this->post_2D.set_tensors_map2(Ds);
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::set_dCs(
	const std::array<std::map<TA, std::map<TAC, Tensor<Tdata>>>,Ndim> &dCs,
	const Tdata_real &threshold_dC,
	const std::string &save_name_suffix)
{
	for(std::size_t ix=0; ix<Ndim; ++ix)
	{
		this->lri.set_tensors_map2( dCs[ix], Label::ab::a, threshold_dC );
		this->lri.set_tensors_map2( dCs[ix], Label::ab::b, threshold_dC );
		this->lri.save_load.save("dCs_"+std::to_string(ix)+"_"+save_name_suffix, {Label::ab::a, Label::ab::b});
	}
	this->flag_finish.dC = true;
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::set_dVs(
	const std::array<std::map<TA, std::map<TAC, Tensor<Tdata>>>,Ndim> &dVs,
	const Tdata_real &threshold_dV,
	const std::string &save_name_suffix)
{
	for(std::size_t ix=0; ix<Ndim; ++ix)
	{
		this->lri.set_tensors_map2( dVs[ix], Label::ab::a0b0, threshold_dV );
		this->lri.save_load.save("dVs_"+std::to_string(ix)+"_"+save_name_suffix, Label::ab::a0b0);
	}
	this->flag_finish.dV = true;
}



template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::cal_Hs(
	const std::array<std::string,3> &save_names_suffix)						// "Cs","Vs","Ds"
{
	assert(this->flag_finish.stru);
	assert(this->flag_finish.C);
	assert(this->flag_finish.V);
	assert(this->flag_finish.D);
	
	this->lri.save_load.load("Cs_"+save_names_suffix[0], {Label::ab::a, Label::ab::b});
	this->lri.save_load.load("Vs_"+save_names_suffix[1], Label::ab::a0b0);
	this->lri.save_load.load("Ds_"+save_names_suffix[2], {Label::ab::a1b1, Label::ab::a1b2, Label::ab::a2b1, Label::ab::a2b2});

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
		this->energy = this->post_2D.cal_energy( this->post_2D.saves["Ds_"+save_names_suffix[2]], Hs_2D );

	this->lri.save_load.save("Cs_"+save_names_suffix[0], {Label::ab::a, Label::ab::b});
	this->lri.save_load.save("Vs_"+save_names_suffix[1], Label::ab::a0b0);
	this->lri.save_load.save("Ds_"+save_names_suffix[2], {Label::ab::a1b1, Label::ab::a1b2, Label::ab::a2b1, Label::ab::a2b2});
}



template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::cal_Fs(
	const std::array<std::string,5> &save_names_suffix)						// "Cs","Vs","Ds","dCs","dVs"	
{
	assert(this->flag_finish.stru);
	assert(this->flag_finish.C);
	assert(this->flag_finish.V);
	assert(this->flag_finish.D);
	assert(this->flag_finish.dC);
	assert(this->flag_finish.dV);

	this->lri.save_load.load("Ds_"+save_names_suffix[2], {Label::ab::a1b1, Label::ab::a1b2, Label::ab::a2b1, Label::ab::a2b2});
	for(std::size_t ix=0; ix<Ndim; ++ix)
	{
		std::map<TA,Tdata> force_ix;

		{
			std::map<TA,std::map<TAC,Tensor<Tdata>>> dHs;

			this->lri.save_load.load("dCs_"+std::to_string(ix)+"_"+save_names_suffix[3], Label::ab::a);
			this->lri.save_load.load("Vs_"+save_names_suffix[1], Label::ab::a0b0);
			this->lri.save_load.load("Cs_"+save_names_suffix[0], Label::ab::b);

			this->lri.coefficient = [](const Label::ab_ab &label, const TA &Aa01, const TAC &Aa2, const TAC &Ab01, const TAC &Ab2) -> Tdata
			{
				switch(label)
				{
					case Label::ab_ab::a0b0_a1b1:	case Label::ab_ab::a0b0_a1b2:	return -1;
					case Label::ab_ab::a0b0_a2b1:	case Label::ab_ab::a0b0_a2b2:	return 1;
					default:	throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
				}
			};

			this->lri.cal({
				Label::ab_ab::a0b0_a1b1,
				Label::ab_ab::a0b0_a1b2,
				Label::ab_ab::a0b0_a2b1,
				Label::ab_ab::a0b0_a2b2},
				dHs);

			this->lri.save_load.save("dCs_"+std::to_string(ix)+"_"+save_names_suffix[3], Label::ab::a);
			this->lri.save_load.save("Vs_"+save_names_suffix[1], Label::ab::a0b0);

			this->lri.save_load.load("Cs_"+save_names_suffix[0], Label::ab::a);
			this->lri.save_load.load("dVs_"+std::to_string(ix)+"_"+save_names_suffix[4], Label::ab::a0b0);

			this->lri.coefficient = nullptr;
			this->lri.cal({
				Label::ab_ab::a0b0_a2b2,
				Label::ab_ab::a0b0_a2b1},
				dHs);

			this->post_2D.cal_force(
				this->post_2D.saves["Ds_"+save_names_suffix[2]],
				this->post_2D.set_tensors_map2(dHs),
				true,
				force_ix );

//			mul(D)
//			this->Fs[ix] = this->post_2D.cal_F(dHs);
//			this->stress[ix] = this->post_2D.cal_sttress(dHs);
	//		this->Fs[ix][I] = \sum_J \sum_{i,j} dHs(i,j) * D(i,j)
		}

		{
			std::map<TA,std::map<TAC,Tensor<Tdata>>> dHs;

			this->lri.coefficient = nullptr;
			this->lri.cal({
				Label::ab_ab::a0b0_a2b2,
				Label::ab_ab::a0b0_a1b2},
				dHs);

			this->lri.save_load.save("dVs_"+std::to_string(ix)+"_"+save_names_suffix[4], Label::ab::a0b0);
			this->lri.save_load.save("Cs_"+save_names_suffix[0], Label::ab::b);

			this->lri.save_load.load("Vs_"+save_names_suffix[1], Label::ab::a0b0);
			this->lri.save_load.load("dCs_"+std::to_string(ix)+"_"+save_names_suffix[3], Label::ab::b);

			this->lri.coefficient = [](const Label::ab_ab &label, const TA &Aa01, const TAC &Aa2, const TAC &Ab01, const TAC &Ab2) -> Tdata
			{
				switch(label)
				{
					case Label::ab_ab::a0b0_a1b1:	case Label::ab_ab::a0b0_a2b1:	return 1;
					case Label::ab_ab::a0b0_a1b2:	case Label::ab_ab::a0b0_a2b2:	return -1;
					default:	throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
				}
			};

			this->lri.cal({
				Label::ab_ab::a0b0_a1b1,
				Label::ab_ab::a0b0_a1b2,
				Label::ab_ab::a0b0_a2b1,
				Label::ab_ab::a0b0_a2b2},
				dHs);

			this->lri.save_load.save("Cs_"+save_names_suffix[0], Label::ab::a);
			this->lri.save_load.save("dCs_"+std::to_string(ix)+"_"+save_names_suffix[3], Label::ab::b);
			this->lri.save_load.save("Vs_"+save_names_suffix[1], Label::ab::a0b0);
			
			this->post_2D.cal_force(
				this->post_2D.saves["Ds_"+save_names_suffix[2]],
				this->post_2D.set_tensors_map2(dHs),
				false,
				force_ix );

//			mul(D)
//			this->Fs[ix] -= this->post_2D.cal_F(dHs);
//			this->stress[ix] -= this->post_2D.cal_sttress(dHs);
	//		this->Fs[ix][J] = \sum_I \sum_{i,j} dHs(i,j) * D(i,j)
		}
		this->force[ix] = this->post_2D.reduce_force(force_ix);
	}
	this->lri.save_load.save("Ds_"+save_names_suffix[2], {Label::ab::a1b1, Label::ab::a1b2, Label::ab::a2b1, Label::ab::a2b2});
}
