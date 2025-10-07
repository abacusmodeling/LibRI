// ===================
//  Author: Peize Lin
//  date: 2023.08.02
// ===================

#pragma once

#include "LRI.h"
#include "LRI_Cal_Aux.h"
#include "../global/Array_Operator.h"
#include "../global/Tensor_Multiply.h"
#include <omp.h>
#include <malloc.h>
#ifdef __MKL_RI
#include <mkl_service.h>
#endif

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
std::map<TA, std::map<std::pair<TA, std::array<Tcell, Ndim>>, std::map<TA, std::map<std::pair<TA, std::array<Tcell, Ndim>>, Tensor<Tdata>>>>>
LRI<TA, Tcell, Ndim, Tdata>::cal_cvc()
{
	using namespace Array_Operator;

	const Data_Pack_Wrapper<TA,TC,Tdata> data_wrapper(this->data_pool, this->data_ab_name);
	const LRI_Cal_Tools<TA,TC,Tdata> tools(this->period, this->data_pool, this->data_ab_name);

  #ifdef __MKL_RI
	const std::size_t mkl_threads = mkl_get_max_threads();
	mkl_set_num_threads(1);
  #endif

	std::map<TA, std::map<TAC, std::map<TA, std::map<TAC, Tensor<Tdata>>>>> cvc;
	/// ???
	std::map<TA, omp_lock_t> lock_cvc_result_add_map = LRI_Cal_Aux::init_lock_result({ Label::ab_ab::a0b0_a2b2 }, this->parallel->list_A, cvc);

	#pragma omp parallel
	{
		std::map<TA, std::map<TAC, std::map<TA, std::map<TAC, Tensor<Tdata>>>>> cvc_thread;
/*	 pseudo code
for I
	for J
		for K
			[CV]_{IK,J}=C^I_{IK}V_{IJ}
			for L
				[CVC]_{IJKL}+=[CV]_{IK,J} C^J_{JL}	#1
	for L
		for K
			[CV]_{IK,L}=C^I_{IK}V_{IL}
			for J
				[CVC]_{IJKL}+=[CV]_{IK,L} C^L_{LJ}	#2
for K
	for J
		for I
			[CV]_{KI,J}=C^K_{KI}V_{KJ}
			for L
				[CVC]_{IJKL}+=[CV]_{KI,J}C^J_{JL}	#3
	for L
		for I
			[CV]_{KI,L}=C^K_{KI}V_{KL}
			for J
				[CVC]_{IJKL}+=[CV]_{KI,L}C^L_{LJ}	#4
*/
		// Main Problems: 
		// How to filter list_I, list_Aa02, list_K, list_L only using the range of Vs?
		// How to deal with thread lock?
		// 转置无法通过改变乘法顺序避免： 输出KL连续，输入KL分别在两个C上
		// for debug
		auto print_a = [](const std::vector<TA>& vec, const std::string name) -> void
			{
				std::cout << name << ": ";
				for (auto& v : vec) { std::cout << v << " "; }
				std::cout << std::endl;
			};
		auto print_ac = [](const std::vector<TAC>& vec, const std::string name) -> void
			{
				std::cout << name << ": ";
				for (auto& v : vec) { std::cout << v.first << " "; }
				std::cout << std::endl;
			};
		const std::vector<TA>  list_I0 = LRI_Cal_Aux::filter_list_map(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).a01, data_wrapper(Label::ab::a).Ds_ab);
		const std::vector<TAC> list_J0 = LRI_Cal_Aux::filter_list_map(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).b01, data_wrapper(Label::ab::b).Ds_ab);
		const std::vector<TAC> list_K = LRI_Cal_Aux::filter_list_set(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).a2, data_wrapper(Label::ab::a).index_Ds_ab[0]);
		const std::vector<TAC> list_L = LRI_Cal_Aux::filter_list_set(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).b2, data_wrapper(Label::ab::b).index_Ds_ab[0]);
		// filter Cs by the range of Vs
		const std::vector<TA>  list_I = LRI_Cal_Aux::filter_list_map(list_I0, data_wrapper(Label::ab::a0b0).Ds_ab);
		const std::vector<TAC> list_J = LRI_Cal_Aux::filter_list_set(list_J0, data_wrapper(Label::ab::a0b0).index_Ds_ab[0]);
		// print_a(list_I0, "list_I0");
		// print_ac(list_J0, "list_J0");
		// print_ac(list_K, "list_K");
		// print_ac(list_L, "list_L");
		// print_a(list_I, "list_I");
		// print_ac(list_J, "list_J");

		// allocate
		// for (TA I : list_I)
		// 	for (TAC J : list_J)
		// 		for (TAC K : list_K)
		// 		{
		// 			const Tensor<Tdata>& C_I_IK = tools.get_Ds_ab(Label::ab::a, I, K);
		// 			const int& ni = C_I_IK.shape[1];
		// 			const int& nk = C_I_IK.shape[2];
		// 			for (TAC L : list_L)
		// 			{
		// 				const TC R_KL = (L.second - K.second) % period;
		// 				const Tensor<Tdata>& C_J_JL = tools.get_Ds_ab(Label::ab::b, J, L);
		// 				const int& nj = C_J_JL.shape[1];
		// 				const int& nl = C_J_JL.shape[2];
		// 				cvc_thread[I][J][K.first][{L.first, R_KL}] = Tensor<Tdata>({ ni, nj, nk, nl });
		// 			}
		// 		}

		// calculate
#pragma omp for schedule(static) collapse(2) nowait
		for (TA I : list_I)
		{
			if (this->filter_atom->filter_for1(Label::ab_ab::a0b0_a2b2, I))	continue; // restrict I in the irreducible sector
			auto& cvc_thread_I = cvc_thread[I];
			for (TAC J : list_J)	 //	term 1
			{
				if (this->filter_atom->filter_for32(Label::ab_ab::a0b0_a2b2, I, J, J))	continue; // restrict (I, J) in the irreducible sector
				const Tensor<Tdata>& V_IJ = tools.get_Ds_ab(Label::ab::a0b0, I, J);
				if (V_IJ.empty()) continue;
				// symmetry: check if (I, J) in irreducible sector
				auto& cvc_thread_IJ = cvc_thread[I][J];
				for (TAC K : list_K)
				{
					const Tensor<Tdata>& C_I_IK = tools.get_Ds_ab(Label::ab::a, I, K);
					if (C_I_IK.empty()) continue;
					// CV_{IK,J}=C^I_{IK}V_{IJ}
					const Tensor<Tdata> CV_IK_J = Tensor_Multiply::x1x2y1_ax1x2_ay1(C_I_IK, V_IJ);
					auto& cvc_thread_IJK = cvc_thread_IJ[K.first];
					for (TAC L : list_L)
					{
						const TC R_KL = (L.second - K.second) % period;
						// std::array<TA, 4> key = { I, J, K, L };
						// [CVC]_{IJKL}+=[CV]_{IK,J}C^J_{JL}	
						const Tensor<Tdata>& C_J_JL = tools.get_Ds_ab(Label::ab::b, J, L);
						if (C_J_JL.empty()) continue;
						//(ika) * (ajl) = (ikjl) -> (ijkl)
						// cvc_thread[{I, J, K, L}][R_KL] += einsum("ika, ajl -> ijkl", CV_IK_J, C_J_JL);
						// cvc_thread[I][J][K.first][{L.first, R_KL}] += Tensor_Multiply::x0x1y1y2_x0x1a_ay1y2(CV_IK_J, C_J_JL).permute_from({ 0,2,1,3 });
						LRI_Cal_Aux::add_Ds(
							Tensor_Multiply::x0x1y1y2_x0x1a_ay1y2(CV_IK_J, C_J_JL).permute_from({ 0,2,1,3 }),
							cvc_thread_IJK[{L.first, R_KL}]);
					}
				}
				LRI_Cal_Aux::add_Ds_omp_try_map(cvc_thread, cvc, lock_cvc_result_add_map, 1.0);
			}
		}
		LRI_Cal_Aux::add_Ds_omp_wait_map(cvc_thread, cvc, lock_cvc_result_add_map, 1.0);

#pragma omp for schedule(static) collapse(2) nowait
		for (TA I : list_I)
		{
			if (this->filter_atom->filter_for1(Label::ab_ab::a0b0_a2b2, I))	continue;
			auto& cvc_thread_I = cvc_thread[I];
			for (TAC L : list_L) // term 2
			{
				const Tensor<Tdata>& V_IL = tools.get_Ds_ab(Label::ab::a0b0, I, L);
				if (V_IL.empty()) continue;
				for (TAC K : list_K)
				{
					const TC& R_KL = (L.second - K.second) % period;
					const Tensor<Tdata>& C_I_IK = tools.get_Ds_ab(Label::ab::a, I, K);
					if (C_I_IK.empty()) continue;
					// CV_{IK,L}: a1a2b0 = a0a1a2 * a0b0
					const Tensor<Tdata> CV_IK_L = Tensor_Multiply::x1x2y1_ax1x2_ay1(C_I_IK, V_IL);
					for (TAC J : list_J)
					{
						if (this->filter_atom->filter_for32(Label::ab_ab::a0b0_a2b2, I, J, J))	continue; // restrict (I, J) in the irreducible sector
						// [CVC]_{IJKL}+=[CV]_{IK,L}C^L_{LJ}
						const Tensor<Tdata>& C_L_LJ = tools.get_Ds_ab(Label::ab::b, L, J);
						if (C_L_LJ.empty()) continue;
						// (ika) * (alj) = (iklj) -> (ijkl)
						// cvc_thread[{I, J, K, L}][R_KL] += einsum("ika, alj -> ijkl", CV_IK_L, C_L_LJ);
						// cvc_thread[I][J][K.first][{L.first, R_KL}] += Tensor_Multiply::x0x1y1y2_x0x1a_ay1y2(CV_IK_L, C_L_LJ).permute_from({ 0,3,1,2 });
						LRI_Cal_Aux::add_Ds(
							Tensor_Multiply::x0x1y1y2_x0x1a_ay1y2(CV_IK_L, C_L_LJ).permute_from({ 0,3,1,2 }),
							cvc_thread_I[J][K.first][{L.first, R_KL}]);
					}
				}
				LRI_Cal_Aux::add_Ds_omp_try_map(cvc_thread, cvc, lock_cvc_result_add_map, 1.0);
			}
		}
		LRI_Cal_Aux::add_Ds_omp_wait_map(cvc_thread, cvc, lock_cvc_result_add_map, 1.0);

#pragma omp for schedule(static) collapse(2) nowait
		for (TAC K : list_K)
		{
			for (TAC J : list_J)	//term 3
			{
				if (this->filter_atom->filter_for1(Label::ab_ab::a0b0_a1b2, J))	continue; // restrict J in the irreducible sector
				const Tensor<Tdata>& V_KJ = tools.get_Ds_ab(Label::ab::a0b0, K, J);
				if (V_KJ.empty()) continue;
				for (TA I : list_I)
				{
					if (this->filter_atom->filter_for32(Label::ab_ab::a0b0_a2b2, I, J, J))	continue; // restrict (I, J) in the irreducible sector
					const Tensor<Tdata>& C_K_KI = tools.get_Ds_ab(Label::ab::a, K, I);
					if (C_K_KI.empty()) continue;
					//[CV]_{KI,J}=C^K_{KI}V_{KJ}
					const Tensor<Tdata> CV_KI_J = Tensor_Multiply::x1x2y1_ax1x2_ay1(C_K_KI, V_KJ);
					auto& cvc_thread_IJK = cvc_thread[I][J][K.first];
					for (TAC L : list_L)
					{
						const TC& R_KL = (L.second - K.second) % period;
						// [CVC]_{IJKL}+=[CV]_{KI,J}C^J_{JL}	#(3)
						const Tensor<Tdata>& C_J_JL = tools.get_Ds_ab(Label::ab::b, J, L);
						if (C_J_JL.empty()) continue;
						// (kia) * (ajl) = (kijl)->(ijkl)
						// cvc_thread[{I, J, K, L}][R_KL] += einsum("kia, ajl -> ijkl", CV_KI_J, C_J_JL);
						// cvc_thread[I][J][K.first][{L.first, R_KL}] += Tensor_Multiply::x0x1y1y2_x0x1a_ay1y2(CV_KI_J, C_J_JL).permute_from({ 1,2,0,3 });
						LRI_Cal_Aux::add_Ds(
							Tensor_Multiply::x0x1y1y2_x0x1a_ay1y2(CV_KI_J, C_J_JL).permute_from({ 1,2,0,3 }),
							cvc_thread_IJK[{L.first, R_KL}]);
					}
				}
				LRI_Cal_Aux::add_Ds_omp_try_map(cvc_thread, cvc, lock_cvc_result_add_map, 1.0);
			}
		}
		LRI_Cal_Aux::add_Ds_omp_wait_map(cvc_thread, cvc, lock_cvc_result_add_map, 1.0);

#pragma omp for schedule(static) collapse(2) nowait
		for (TAC K : list_K)
		{
			for (TAC L : list_L) // term 4
			{
				const Tensor<Tdata>& V_KL = tools.get_Ds_ab(Label::ab::a0b0, K, L);
				if (V_KL.empty()) continue;
				const TC& R_KL = (L.second - K.second) % period;
				for (TA I : list_I)
				{
					if (this->filter_atom->filter_for1(Label::ab_ab::a0b0_a2b2, I))	continue; // restrict I in the irreducible sector
					const Tensor<Tdata>& C_K_KI = tools.get_Ds_ab(Label::ab::a, K, I);
					if (C_K_KI.empty()) continue;
					// CV_{IK,L}: a1a2b0 = a0a1a2 * a0b0
					const Tensor<Tdata> CV_KI_L = Tensor_Multiply::x1x2y1_ax1x2_ay1(C_K_KI, V_KL);
					for (TAC J : list_J)
					{
						if (this->filter_atom->filter_for32(Label::ab_ab::a0b0_a2b2, I, J, J))	continue; // restrict (I, J) in the irreducible sector
						// [CVC]_{IJKL}+=[CV]_{IK,L}C^L_{LJ}
						const Tensor<Tdata>& C_L_LJ = tools.get_Ds_ab(Label::ab::b, L, J);
						if (C_L_LJ.empty()) continue;
						// (kia) * (alj) = (kilj) -> (ijkl)
						// cvc_thread[{I, J, K, L}][R_KL] += einsum("kia, alj -> ijkl", CV_KI_L, C_L_LJ);
						// cvc_thread[I][J][K.first][{L.first, R_KL}] += Tensor_Multiply::x0x1y1y2_x0x1a_ay1y2(CV_KI_L, C_L_LJ).permute_from({ 1,3,0,2 });
						LRI_Cal_Aux::add_Ds(
							Tensor_Multiply::x0x1y1y2_x0x1a_ay1y2(CV_KI_L, C_L_LJ).permute_from({ 1,3,0,2 }),
							cvc_thread[I][J][K.first][{L.first, R_KL}]);
					}
				}
				LRI_Cal_Aux::add_Ds_omp_try_map(cvc_thread, cvc, lock_cvc_result_add_map, 1.0);
			}
		}
		LRI_Cal_Aux::add_Ds_omp_wait_map(cvc_thread, cvc, lock_cvc_result_add_map, 1.0);
	} // end #pragma omp parallel

	LRI_Cal_Aux::destroy_lock_result(lock_cvc_result_add_map, cvc);

  #ifdef __MKL_RI
	mkl_set_num_threads(mkl_threads);
  #endif

	malloc_trim(0);
	return cvc;
}	// end LRI::cal_cvc

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
std::map<TA, std::map<std::pair<TA, std::array<Tcell, Ndim>>, Tensor<Tdata>>>
LRI<TA, Tcell, Ndim, Tdata>::constract_cvc_ds(
	const std::map<TA, std::map<TAC, std::map<TA, std::map<TAC, Tensor<Tdata>>>>>& cvc)
{
	const Data_Pack_Wrapper<TA, TC, Tdata> data_wrapper(this->data_pool, this->data_ab_name);
	const LRI_Cal_Tools<TA, TC, Tdata> tools(this->period, this->data_pool, this->data_ab_name);

	const std::vector<TA>  list_I = LRI_Cal_Aux::filter_list_map(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).a01, data_wrapper(Label::ab::a).Ds_ab);
	const std::vector<TAC> list_J = LRI_Cal_Aux::filter_list_map(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).b01, data_wrapper(Label::ab::b).Ds_ab);
	const std::vector<TAC> list_K0 = LRI_Cal_Aux::filter_list_set(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).a2, data_wrapper(Label::ab::a).index_Ds_ab[0]);
	const std::vector<TAC> list_L0 = LRI_Cal_Aux::filter_list_set(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).b2, data_wrapper(Label::ab::b).index_Ds_ab[0]);

	const std::vector<TAC> list_K = LRI_Cal_Aux::filter_list_map(list_K0, data_wrapper(Label::ab::a2b2).Ds_ab);
	const std::vector<TAC> list_L = LRI_Cal_Aux::filter_list_set(list_L0, data_wrapper(Label::ab::a2b2).index_Ds_ab[0]);

	std::map<TA, std::map<TAC, Tensor<Tdata>>> Hs;
	for (auto& map_i : cvc)
	{
		const TA& I = map_i.first;
		if (this->filter_atom->filter_for1(Label::ab_ab::a0b0_a2b2, I))	continue; // restrict I in the irreducible sector
		for (auto& map_j : map_i.second)
		{
			const TAC& J = map_j.first;
			if (this->filter_atom->filter_for32(Label::ab_ab::a0b0_a2b2, I, J, J))	continue; // restrict (I, J) in the irreducible sector
			// init H_IJ
			for (auto& map_k : map_j.second)
			{
				const TA& K = map_k.first;
				for (auto& map_l : map_k.second)
				{
					const TAC& L = map_l.first;
					const Tensor<Tdata>& CVC_IJKL = map_l.second;
					const Tensor<Tdata>& D_KL = tools.get_Ds_ab(Label::ab::a2b2, K, L);
					if (D_KL.empty()) continue;
					// Hs[I][J] += Tensor_Multiply::gemv(CVC_IJKL, D_KL)
					LRI_Cal_Aux::add_Ds(Tensor_Multiply::gemv(CVC_IJKL, D_KL), Hs[I][J]);
				}
			}
		}
	}
	return Hs;
}

}	// end namespace RI

