// ===================
//  Author: Ziqing Guan
//  date: 2025.12.26
// ===================

#pragma once
#include "../global/Global_Func-2.h"
#include "../global/Tensor.h"
#include "../global/Global_Func-1.h"
#include "../ri/LRI_k.h"

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
	using Tk = std::array<double,Ndim>;

	Hartree () = default;

	void init(std::vector<Tk> kindex_map_in)
	{
		this->kindex_map = std::move(kindex_map_in);
	}

	void set_parallel(
		const MPI_Comm &mpi_comm_in, const std::size_t nat, const std::size_t nk,
		const std::array<Tcell, Ndim> &period_in)
    {
        this->lrik.mpi_comm = mpi_comm_in;
		this->lrik.period = period_in;
		RI::Distribute_Equally::distribute_atom_pair_and_k(mpi_comm_in,
			nat, nk, this->list_I, this->list_J, this->k_indices, false);

		this->list_IJ = Global_Func::set_union(this->list_I, this->list_J);
		this->flag_finish.stru = true;
    }

	void set_Cs(
		std::map<TA, std::map<TAC, Tensor<Tdata>>> &Cs,
		const Tdata_real &threshold,
		const std::set<TA> &listI,
		const std::set<TA> &listJ,
		const std::string &save_name_suffix="")
	{
		Cs = Communicate_Tensors_Map_Judge::comm_map2_first(this->lrik.mpi_comm, std::move(Cs), listI, listJ);
		this->lrik.set_tensors_map2(
			Cs,
			{Label::ab::a, Label::ab::b},
			{{"threshold_filter", threshold}, {"flag_comm", false}},
			"Cs_"+save_name_suffix );
		this->flag_finish.Cs = true;
	}

	void free_Cs(const std::string &save_name_suffix="")
	{
		this->lrik.free_tensors_map2("Cs_"+save_name_suffix);
		this->flag_finish.Cs = false;
	};

	void set_Vs(
		std::map<TA, std::map<TAC, Tensor<Tdata>>> &Vs,
		const Tdata_real &threshold,
		const std::set<TA> &listI,
		const std::set<TA> &listJ,
		const std::string &save_name_suffix="")
	{
		Vs = Communicate_Tensors_Map_Judge::comm_map2_first(this->lrik.mpi_comm, std::move(Vs), listI, listJ);
		this->lrik.set_tensors_map2(
			Vs,
			{Label::ab::a0b0},
			{{"threshold_filter", threshold}, {"flag_comm", false}},
			"Vs_"+save_name_suffix );
		this->flag_finish.Vs = true;
	};

    void free_Vs(const std::string &save_name_suffix="")
	{
		this->lrik.free_tensors_map2("Vs_"+save_name_suffix);
		this->flag_finish.Vs = false;
	};

	std::map<TA, std::map<TA, std::map<int, Tensor<Tdata>>>> cal_hartree(
		const std::map<TA, std::map<TA, std::map<int, Tensor<Tdata>>>>& Ds,
		const std::string &save_name_C="Cs_", const std::string &save_name_V="Vs_")
	{
		return this->lrik.cal_cvcd_k_hartree(
			Ds,	this->kindex_map, this->k_indices, this->list_I, this->list_J, this->list_IJ,
			save_name_C, save_name_V);
	}

	std::vector<Tk> kindex_map;		// index → Tk fractional coord
	std::vector<int> k_indices;
	std::vector<TA> list_I;
	std::vector<TA> list_J;
	std::vector<TA> list_IJ;		// I∪J

	LRI_k<TA,Tcell,Ndim,Tdata> lrik;

private:
	struct Flag_Finish
	{
		bool stru=false;
		bool Cs=false;
		bool Vs=false;
	};
	Flag_Finish flag_finish;
	
};

}
