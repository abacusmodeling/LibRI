// ===================
//  Author: Ziqing Guan
//  date: 2026.08.02
// ===================

#pragma once

#include "LRI.h"

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
class LRI_k : public LRI<TA, Tcell, Ndim, Tdata>
{
public:
    using TC = std::array<Tcell,Ndim>;
    using TAC = std::pair<TA,TC>;
    using Tdata_real = Global_Func::To_Real_t<Tdata>;
	using Tk = std::array<double, Ndim>;

	LRI_k() = default;

    std::map<int, std::map<TA, Tensor<Tdata>>> cal_Csk_ao_mo(
        const std::map<TA, std::map<TAC, Tensor<Tdata>>>& CsR_ao,
        const std::map<int, std::map<TA, Tensor<Tdata>>>& map_psi,
        const std::vector<Tk>& kindex_map,
        const std::vector<int>& k_indices, const std::vector<TA>& list_IJ, std::ofstream& ofs);

    std::map<int, std::map<int, Tensor<Tdata>>> cal_cvc_mo_k_onthefly(
        const std::map<int, std::map<TA, Tensor<Tdata>>>& Cs_ao_mo,
        const std::map<int, std::map<TA, Tensor<Tdata>>>& map_psi,
        const std::vector<int>& k1_indices,
        const std::vector<int>& k2_indices,
        const std::vector<TA>& list_I,
        const std::vector<TA>& list_J,
        const std::vector<std::string>& psi_type,
        const std::size_t nocc,
        const std::size_t nvirt,
        const std::string& save_name,
        const bool is_A,
        const std::vector<Tk>& q_list_in,
        const std::map<Tk,std::vector<std::pair<int, int>>>& q2kpair_in);

    std::map<int, std::map<int, Tensor<Tdata>>> cal_cvc_mo_k_hartree_onthefly(
        const std::map<int, std::map<TA, Tensor<Tdata>>>& Cs_ao_mo,
        const std::map<int, std::map<TA, Tensor<Tdata>>>& map_psi,
        const std::vector<int>& k1_indices,
        const std::vector<int>& k2_indices,
        const std::vector<TA>& list_I,
        const std::vector<TA>& list_J,
        const std::vector<std::string>& psi_type,
        const std::size_t nocc,
        const std::size_t nvirt,
        const std::string& save_name,
        const bool is_A);

    std::map<TA, std::map<TA, std::map<int, Tensor<Tdata>>>> cal_cvcd_k_hartree(
        const std::map<TA, std::map<TA, std::map<int, Tensor<Tdata>>>>& Ds,  // D(s,t)[k]
        const std::vector<Tk>& kindex_map,// k index to direct coordinate array<double, Ndim>
        const std::vector<int>& list_k_index,
        const std::vector<TA>& list_I,
        const std::vector<TA>& list_J,
        const std::vector<TA>& list_IJ,
		const std::string& save_name_C,
        const std::string& save_name_V);

};
}

#include "LRI_k-cal_cvc_mo.hpp"
#include "LRI_k-cal_hartree.hpp"
