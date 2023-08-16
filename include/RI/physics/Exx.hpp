// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "Exx.h"
#include "../ri/Cell_Nearest.h"
#include "../ri/Label.h"

#include <cassert>

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::set_parallel(
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
		this->post_2D.set_parallel(this->mpi_comm, this->atoms_pos, this->period);
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
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

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::set_Vs(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Vs,
	const Tdata_real &threshold_V,
	const std::string &save_name_suffix)
{
	this->lri.set_tensors_map2( Vs, Label::ab::a0b0, threshold_V );
	this->lri.save_load.save("Vs_"+save_name_suffix, Label::ab::a0b0);
	this->flag_finish.V = true;
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
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
	const std::array<std::map<TA, std::map<TAC, Tensor<Tdata>>>,Npos> &dCs,
	const Tdata_real &threshold_dC,
	const std::string &save_name_suffix)
{
	for(std::size_t ipos=0; ipos<Npos; ++ipos)
	{
		this->lri.set_tensors_map2( dCs[ipos], Label::ab::a, threshold_dC );
		this->lri.set_tensors_map2( dCs[ipos], Label::ab::b, threshold_dC );
		this->lri.save_load.save("dCs_"+std::to_string(ipos)+"_"+save_name_suffix, {Label::ab::a, Label::ab::b});
	}
	this->flag_finish.dC = true;
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::set_dVs(
	const std::array<std::map<TA, std::map<TAC, Tensor<Tdata>>>,Npos> &dVs,
	const Tdata_real &threshold_dV,
	const std::string &save_name_suffix)
{
	for(std::size_t ipos=0; ipos<Npos; ++ipos)
	{
		this->lri.set_tensors_map2( dVs[ipos], Label::ab::a0b0, threshold_dV );
		this->lri.save_load.save("dVs_"+std::to_string(ipos)+"_"+save_name_suffix, Label::ab::a0b0);
	}
	this->flag_finish.dV = true;
}



template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
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

	std::vector<std::map<TA, std::map<TAC, Tensor<Tdata>>>> Hs_vec(1);
	this->lri.coefficients = {nullptr};
	this->lri.cal_loop3(
		{Label::ab_ab::a0b0_a1b1,
		 Label::ab_ab::a0b0_a1b2,
		 Label::ab_ab::a0b0_a2b1,
		 Label::ab_ab::a0b0_a2b2},
		Hs_vec);
	this->Hs = std::move(Hs_vec[0]);

	//if()
		this->energy = this->post_2D.cal_energy(
			this->post_2D.saves["Ds_"+save_names_suffix[2]],
			this->post_2D.set_tensors_map2(this->Hs) );

	this->lri.save_load.save("Cs_"+save_names_suffix[0], {Label::ab::a, Label::ab::b});
	this->lri.save_load.save("Vs_"+save_names_suffix[1], Label::ab::a0b0);
	this->lri.save_load.save("Ds_"+save_names_suffix[2], {Label::ab::a1b1, Label::ab::a1b2, Label::ab::a2b1, Label::ab::a2b2});
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::cal_force(
	const std::array<std::string,5> &save_names_suffix)						// "Cs","Vs","Ds","dCs","dVs"
{
	assert(this->flag_finish.stru);
	assert(this->flag_finish.C);
	assert(this->flag_finish.V);
	assert(this->flag_finish.D);
	assert(this->flag_finish.dC);
	assert(this->flag_finish.dV);

	this->lri.save_load.load("Ds_"+save_names_suffix[2], {Label::ab::a1b1, Label::ab::a1b2, Label::ab::a2b1, Label::ab::a2b2});
	for(std::size_t ipos=0; ipos<Npos; ++ipos)
	{
		std::map<TA,Tdata> force_ipos;

		{
			std::vector<std::map<TA,std::map<TAC,Tensor<Tdata>>>> dHs_vec(1);

			this->lri.save_load.load("dCs_"+std::to_string(ipos)+"_"+save_names_suffix[3], Label::ab::a);
			this->lri.save_load.load("Vs_"+save_names_suffix[1], Label::ab::a0b0);
			this->lri.save_load.load("Cs_"+save_names_suffix[0], Label::ab::b);

			this->lri.cal_loop3(
				{Label::ab_ab::a0b0_a1b1,
				 Label::ab_ab::a0b0_a1b2,},
				dHs_vec,
				-1.0);

			this->lri.cal_loop3(
				{Label::ab_ab::a0b0_a2b1,
				 Label::ab_ab::a0b0_a2b2},
				dHs_vec,
				1.0);

			this->lri.save_load.save("dCs_"+std::to_string(ipos)+"_"+save_names_suffix[3], Label::ab::a);
			this->lri.save_load.save("Vs_"+save_names_suffix[1], Label::ab::a0b0);

			this->lri.save_load.load("Cs_"+save_names_suffix[0], Label::ab::a);
			this->lri.save_load.load("dVs_"+std::to_string(ipos)+"_"+save_names_suffix[4], Label::ab::a0b0);

			this->lri.cal_loop3(
				{Label::ab_ab::a0b0_a2b2,
				 Label::ab_ab::a0b0_a2b1},
				dHs_vec,
				1.0);

			this->post_2D.cal_force(
				this->post_2D.saves["Ds_"+save_names_suffix[2]],
				this->post_2D.set_tensors_map2(std::move(dHs_vec[0])),
				true,
				force_ipos );

//			mul(D)
//			this->Fs[ipos] = this->post_2D.cal_F(dHs);
//			this->stress[ipos] = this->post_2D.cal_sttress(dHs);
	//		this->Fs[ipos][I] = \sum_J \sum_{i,j} dHs(i,j) * D(i,j)
		}

		{
			std::vector<std::map<TA,std::map<TAC,Tensor<Tdata>>>> dHs_vec(1);

			this->lri.cal_loop3(
				{Label::ab_ab::a0b0_a2b2,
				 Label::ab_ab::a0b0_a1b2},
				dHs_vec,
				1.0);

			this->lri.save_load.save("dVs_"+std::to_string(ipos)+"_"+save_names_suffix[4], Label::ab::a0b0);
			this->lri.save_load.save("Cs_"+save_names_suffix[0], Label::ab::b);

			this->lri.save_load.load("Vs_"+save_names_suffix[1], Label::ab::a0b0);
			this->lri.save_load.load("dCs_"+std::to_string(ipos)+"_"+save_names_suffix[3], Label::ab::b);

			this->lri.cal_loop3(
				{Label::ab_ab::a0b0_a1b1,
				 Label::ab_ab::a0b0_a2b1},
				dHs_vec,
				1.0);

			this->lri.cal_loop3(
				{Label::ab_ab::a0b0_a1b2,
				 Label::ab_ab::a0b0_a2b2},
				dHs_vec,
				-1.0);

			this->lri.save_load.save("Cs_"+save_names_suffix[0], Label::ab::a);
			this->lri.save_load.save("dCs_"+std::to_string(ipos)+"_"+save_names_suffix[3], Label::ab::b);
			this->lri.save_load.save("Vs_"+save_names_suffix[1], Label::ab::a0b0);

			this->post_2D.cal_force(
				this->post_2D.saves["Ds_"+save_names_suffix[2]],
				this->post_2D.set_tensors_map2(std::move(dHs_vec[0])),
				false,
				force_ipos );

//			mul(D)
//			this->Fs[ipos] -= this->post_2D.cal_F(dHs);
//			this->stress[ipos] -= this->post_2D.cal_sttress(dHs);
	//		this->Fs[ipos][J] = \sum_I \sum_{i,j} dHs(i,j) * D(i,j)
		}
		this->force[ipos] = this->post_2D.reduce_force(force_ipos);
	} // end for(ipos)
	this->lri.save_load.save("Ds_"+save_names_suffix[2], {Label::ab::a1b1, Label::ab::a1b2, Label::ab::a2b1, Label::ab::a2b2});
}


template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void Exx<TA,Tcell,Ndim,Tdata>::cal_stress(
	const std::array<std::string,5> &save_names_suffix)						// "Cs","Vs","Ds","dCs","dVs"
{
	assert(this->flag_finish.stru);
	assert(this->flag_finish.C);
	assert(this->flag_finish.V);
	assert(this->flag_finish.D);
	assert(this->flag_finish.dC);
	assert(this->flag_finish.dV);

	Cell_Nearest<TA,Tcell,Ndim,Tpos,Npos> cell_nearest;
	cell_nearest.init(this->atoms_pos, this->latvec, this->period);

	using namespace Array_Operator;
	auto get_delta_pos = [this, &cell_nearest](
		const TA &Ax, const TA &Ay, const std::array<Tcell,Ndim> &celly, const std::size_t &ipos1)
		-> Tpos
	{
		Tpos delta_pos = this->atoms_pos[Ay][ipos1] - this->atoms_pos[Ax][ipos1];
		const std::array<Tcell,Ndim> celly_nearest = cell_nearest.get_cell_nearest_discrete(Ax, Ay, celly);
		for(std::size_t idim=0; idim<Ndim; ++idim)
			delta_pos += celly_nearest[idim] * this->latvec[idim][ipos1];
		return delta_pos;
	};

	this->stress = Tensor<Tdata>({Npos, Npos});

	this->lri.save_load.load("Ds_"+save_names_suffix[2], {Label::ab::a1b1, Label::ab::a2b1});
	for(std::size_t ipos0=0; ipos0<Npos; ++ipos0)
	{
		std::vector<std::map<TA,std::map<TAC,Tensor<Tdata>>>> dHs_vec(Npos);
		this->lri.coefficients.resize(Npos);

		this->lri.save_load.load("dCs_"+std::to_string(ipos0)+"_"+save_names_suffix[3], Label::ab::a);
		this->lri.save_load.load("Vs_"+save_names_suffix[1], Label::ab::a0b0);
		this->lri.save_load.load("Cs_"+save_names_suffix[0], Label::ab::b);

		for(std::size_t ipos1=0; ipos1<Npos; ++ipos1)
			this->lri.coefficients[ipos1] =
				[this,ipos1,&get_delta_pos](
					const Label::ab_ab &label, const TA &Aa01, const TAC &Aa2, const TAC &Ab01, const TAC &Ab2) -> Tdata
				{	return get_delta_pos(Aa01, Aa2.first, Aa2.second, ipos1);	};

		this->lri.cal(
			{Label::ab_ab::a0b0_a1b1,
			 Label::ab_ab::a0b0_a2b1},
			dHs_vec);

		this->lri.save_load.save("dCs_"+std::to_string(ipos0)+"_"+save_names_suffix[3], Label::ab::a);
		this->lri.save_load.save("Vs_"+save_names_suffix[1], Label::ab::a0b0);

		this->lri.save_load.load("Cs_"+save_names_suffix[0], Label::ab::a);
		this->lri.save_load.load("dVs_"+std::to_string(ipos0)+"_"+save_names_suffix[4], Label::ab::a0b0);

		for(std::size_t ipos1=0; ipos1<Npos; ++ipos1)
			this->lri.coefficients[ipos1] =
				[this,ipos1,&get_delta_pos](
					const Label::ab_ab &label, const TA &Aa01, const TAC &Aa2, const TAC &Ab01, const TAC &Ab2) -> Tdata
				{	return get_delta_pos(Aa01, Ab01.first, Ab01.second, ipos1);	};

		this->lri.cal(
			{Label::ab_ab::a0b0_a1b1,
			 Label::ab_ab::a0b0_a2b1},
			dHs_vec);

		this->lri.save_load.save("dVs_"+std::to_string(ipos0)+"_"+save_names_suffix[4], Label::ab::a0b0);
		this->lri.save_load.save("Cs_"+save_names_suffix[0], Label::ab::b);

		this->lri.save_load.load("Vs_"+save_names_suffix[1], Label::ab::a0b0);
		this->lri.save_load.load("dCs_"+std::to_string(ipos0)+"_"+save_names_suffix[3], Label::ab::b);

		for(std::size_t ipos1=0; ipos1<Npos; ++ipos1)
			this->lri.coefficients[ipos1] =
				[this,ipos1,&get_delta_pos](
					const Label::ab_ab &label, const TA &Aa01, const TAC &Aa2, const TAC &Ab01, const TAC &Ab2) -> Tdata
				{	return get_delta_pos(Ab01.first, Ab2.first, (Ab2.second-Ab01.second)%this->period, ipos1); };

		this->lri.cal(
			{Label::ab_ab::a0b0_a1b1,
			 Label::ab_ab::a0b0_a2b1},
			dHs_vec);

		this->lri.save_load.save("Cs_"+save_names_suffix[0], Label::ab::a);
		this->lri.save_load.save("Vs_"+save_names_suffix[1], Label::ab::a0b0);
		this->lri.save_load.save("dCs_"+std::to_string(ipos0)+"_"+save_names_suffix[3], Label::ab::b);

		for(std::size_t ipos1=0; ipos1<Npos; ++ipos1)
			this->stress(ipos0,ipos1) = post_2D.cal_energy(
				this->post_2D.saves["Ds_"+save_names_suffix[2]],
				this->post_2D.set_tensors_map2(dHs_vec[ipos1]));
	}
	this->lri.save_load.save("Ds_"+save_names_suffix[2], {Label::ab::a1b1, Label::ab::a2b1});
}


}