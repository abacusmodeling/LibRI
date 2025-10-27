// ===================
//  Author: Ziqing Guan
//  date: 2025.12.26
// ===================

#pragma once
#include "../global/Global_Func-2.h"
#include "../global/Tensor.h"
#include "../ri/LRI.h"

#include <mpi.h>
#include <array>
#include <map>
#include <set>

namespace RI
{
// feat: calculate hartree term for qs-GW based on cvcd in k space
template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
class Hartree
{
public:
	using TC = std::array<Tcell,Ndim>;
	using TAC = std::pair<TA,TC>;
	using Tdata_real = Global_Func::To_Real_t<Tdata>;
	using Tatom_pos = std::array<double,Ndim>;		// tmp

	void set_parallel(
		const MPI_Comm &mpi_comm,
		const std::map<TA,Tatom_pos> &atoms_pos,
		const std::array<Tatom_pos,Ndim> &latvec,
		const std::array<Tcell,Ndim> &period)
    {
        this->lri.set_parallel(
            mpi_comm, atoms_pos, latvec, period,
            {}/*no label*/);
        this->flag_finish.stru = true;
    }

	void set_Cs(
		std::map<TA, std::map<TAC, Tensor<Tdata>>> &Cs,
		const Tdata_real &threshold,
		const std::set<TA> &listI,
		const std::set<TA> &listJ,
		const std::string &save_name_suffix="")
	{
		Cs = Communicate_Tensors_Map_Judge::comm_map2_first(this->lri.mpi_comm, std::move(Cs), listI, listJ);
		this->lri.set_tensors_map2(
			Cs,
			{Label::ab::a, Label::ab::b},
			{{"threshold_filter", threshold}, {"flag_comm", false}},
			"Cs_"+save_name_suffix );
		this->flag_finish.Cs = true;
	}

	void free_Cs(const std::string &save_name_suffix="")
	{
		this->lri.free_tensors_map2("Cs_"+save_name_suffix);
		this->flag_finish.Cs = false;
	};

	void set_Vs(
		std::map<TA, std::map<TAC, Tensor<Tdata>>> &Vs,
		const Tdata_real &threshold,
		const std::set<TA> &listI,
		const std::set<TA> &listJ,
		const std::string &save_name_suffix="")
	{
		Vs = Communicate_Tensors_Map_Judge::comm_map2_first(this->lri.mpi_comm, std::move(Vs), listI, listJ);
		this->lri.set_tensors_map2(
			Vs,
			{Label::ab::a0b0},
			{{"threshold_filter", threshold}, {"flag_comm", false}},
			"Vs_"+save_name_suffix );
		this->flag_finish.Vs = true;
	};

    void free_Vs(const std::string &save_name_suffix="")
	{
		this->lri.free_tensors_map2("Vs_"+save_name_suffix);
		this->flag_finish.Vs = false;
	};



public:
	LRI<TA,Tcell,Ndim,Tdata> lri;

	struct Flag_Finish
	{
		bool stru=false;
		bool Cs=false;
		bool Vs=false;
		bool Ds=false;
	};
	Flag_Finish flag_finish;
	
};

}