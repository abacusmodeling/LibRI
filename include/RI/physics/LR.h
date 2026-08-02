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
template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
class LR
{
public:
	using TC = std::array<Tcell,Ndim>;
	using TAC = std::pair<TA,TC>;
	using Tdata_real = Global_Func::To_Real_t<Tdata>;
	using Tk = std::array<double,Ndim>;

	LR() = default;

	void init(std::vector<Tk> kindex_map_in, int nocc_in, int nvirt_in)
	{
		this->kindex_map = std::move(kindex_map_in);
		this->nocc = nocc_in;
		this->nvirt = nvirt_in;
	}

	void set_parallel(
		const MPI_Comm &mpi_comm_in, const std::size_t nat, const std::size_t nk,
		const std::array<Tcell, Ndim> &period_in)
	{
        this->lrik.mpi_comm = mpi_comm_in;
		this->lrik.period = period_in;

		RI::Distribute_Equally::distribute_atom_and_k_pair(mpi_comm_in,
			nat, nk, this->list_I, this->list_J,
			this->k1_indices, this->k2_indices, false);

		this->list_IJ = Global_Func::set_union(this->list_I, this->list_J);
		this->k_indices = Global_Func::set_union(this->k1_indices, this->k2_indices);

        this->flag_finish.stru = true;
    }

	void set_Cs(
		std::map<TA, std::map<TAC, Tensor<Tdata>>> &Cs,
		const Tdata_real &threshold,
		const std::set<TA> &listIJ,
		const std::set<TA> &all_atoms,
		const std::string &save_name_suffix="")
	{
		//Cs = Communicate_Tensors_Map_Judge::comm_map2_first(this->lrik.mpi_comm, std::move(Cs), listI, listJ);
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
		//Vs = Communicate_Tensors_Map_Judge::comm_map2_first(this->lrik.mpi_comm, std::move(Vs), listI, listJ);
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

	void set_Ws(
		std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ws,
		const Tdata_real &threshold,
		const std::set<TA> &listI,
		const std::set<TA> &listJ,
		const std::string &save_name_suffix="")
	{
		//Ws = Communicate_Tensors_Map_Judge::comm_map2_first(this->lrik.mpi_comm, std::move(Ws), listI, listJ);
		this->lrik.set_tensors_map2(
			Ws,
			{Label::ab::a0b0},
			{{"threshold_filter", threshold}, {"flag_comm", false}},
			"Ws_"+save_name_suffix );
		this->flag_finish.Ws = true;
	};
	void free_Ws(const std::string &save_name_suffix="")
	{
		this->lrik.free_tensors_map2("Ws_"+save_name_suffix);
		this->flag_finish.Ws = false;
	};

	/// @brief calculate Csk_ao_mo on-the-fly and store internally
	void cal_Csk_ao_mo(
		const std::string& save_name,
		std::ofstream& ofs)
	{
		const auto& CsR_ao = this->lrik.data_pool.at(save_name).Ds_ab;
		this->Csk_ao_mo = this->lrik.cal_Csk_ao_mo(
			CsR_ao, this->map_psi, this->kindex_map,
			this->k_indices, this->list_IJ, ofs);
	}

	std::map<int, std::map<int, Tensor<Tdata>>>
	cal_cvc_mo_k_onthefly(
		const std::vector<std::string>& psi_type,
		const std::string& save_name,
		const bool is_A)
	{
		return this->lrik.cal_cvc_mo_k_onthefly(
			this->Csk_ao_mo, this->map_psi,
			this->k1_indices, this->k2_indices,
			this->list_I, this->list_J,
			psi_type, this->nocc, this->nvirt,
			save_name, is_A,
			this->q_list, this->q2kpair);
	}

	std::map<int, std::map<int, Tensor<Tdata>>>
	cal_cvc_mo_k_hartree_onthefly(
		const std::vector<std::string>& psi_type,
		const std::string& save_name,
		const bool is_A)
	{
		return this->lrik.cal_cvc_mo_k_hartree_onthefly(
			this->Csk_ao_mo, this->map_psi,
			this->k1_indices, this->k2_indices,
			this->list_I, this->list_J,
			psi_type, this->nocc, this->nvirt,
			save_name, is_A);
	}

	std::vector<Tk> kindex_map;		// index → Tk fractional coord
	std::size_t nocc;
	std::size_t nvirt;

	std::vector<int> k1_indices;
	std::vector<int> k2_indices;
	std::vector<int> k_indices;		// k1∪k2 (for psi/mo transform)
	std::vector<TA> list_I;
	std::vector<TA> list_J;
	std::vector<TA> list_IJ;		// I∪J

	std::vector<Tk> q_list;
	std::map<Tk, std::vector<std::pair<int, int>>> q2kpair; // q → list of (k1_idx, k2_idx)

	std::map<int, std::map<TA, Tensor<Tdata>>> map_psi;		//<k, <iat, tensor{nmo, iat.nw}>>
	std::map<int, std::map<TA, Tensor<Tdata>>> Csk_ao_mo;
	LRI_k<TA,Tcell,Ndim,Tdata> lrik;

private:
	struct Flag_Finish
	{
		bool stru=false;
		bool Cs=false;
		bool Vs=false;
		bool Ws=false;
	};
	Flag_Finish flag_finish;

};

}
