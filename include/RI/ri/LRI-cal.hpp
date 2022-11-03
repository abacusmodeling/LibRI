// ===================
//  Author: Peize Lin
//  date: 2022.08.12
// ===================

#pragma once

#include "LRI.h"
#include "LRI_Cal_Aux.h"

#include <omp.h>
#ifdef __MKL
#include <mkl_service.h>
#endif

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void LRI<TA,Tcell,Ndim,Tdata>::cal(
	const std::vector<Label::ab_ab> &labels,
	std::vector<std::map<TA, std::map<TAC, Tensor<Tdata>>>> &Ds_result)
{
	using namespace Array_Operator;

	const bool flag_D_b_transpose = [&labels]() -> bool
	{
		for(const Label::ab_ab &label : labels)
			if(LRI_Cal_Aux::judge_x(label)==0)
				return true;
		return false;
	}();

	assert(!this->coefficients.empty());
	if(Ds_result.empty())
		Ds_result.resize(this->coefficients.size());
	else
		assert(Ds_result.size()==this->coefficients.size());

	omp_lock_t lock_Ds_result_add;
	omp_init_lock(&lock_Ds_result_add);	

#ifdef __MKL
    const size_t mkl_threads = mkl_get_max_threads();
//	if(!omp_get_nested())
//		mkl_set_num_threads(std::max(1UL,mkl_threads/list_Aa01.size()));
//	else
		mkl_set_num_threads(1);
#endif

	#pragma omp parallel
	{
		std::vector<std::map<TA, std::map<TAC, Tensor<Tdata>>>> Ds_result_thread(this->coefficients.size());
		LRI_Cal_Tools<TA,TC,Tdata> tools(this->period, this->Ds_ab, Ds_result_thread);

		const std::vector<TA> &list_Aa01 = this->parallel->get_list_Aa01();
		for(size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
		{
			const TA &Aa01 = list_Aa01[ia01];

			const std::vector<TAC> &list_Aa2 = this->parallel->get_list_Aa2(Aa01);
			for(size_t ia2=0; ia2<list_Aa2.size(); ++ia2)
			{
				const TAC &Aa2 = list_Aa2[ia2];
				const Tensor<Tdata> D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
				if(D_a.empty())	continue;

				const std::vector<TAC> &list_Ab01 = this->parallel->get_list_Ab01(Aa01, Aa2);
				#pragma omp for schedule(dynamic) nowait
				for(size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
				{
					const TAC &Ab01 = list_Ab01[ib01];
					std::unordered_map<Label::ab_ab, Tensor<Tdata>> Ds_b01;
					std::unordered_map<Label::ab_ab, Tdata_real> Ds_b01_csm;
					Ds_b01.reserve(Label::array_ab_ab.size());
					Ds_b01_csm.reserve(Label::array_ab_ab.size());

					const std::vector<TAC> &list_Ab2 = this->parallel->get_list_Ab2(Aa01, Aa2, Ab01);
					for(size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
					{
						const TAC &Ab2 = list_Ab2[ib2];
						const Tensor<Tdata> D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
						if(D_b.empty())	continue;
						const Tensor<Tdata> D_b_transpose =
							flag_D_b_transpose ? LRI_Cal_Aux::tensor3_transpose(D_b) : Tensor<Tdata>{};

						for(const Label::ab_ab &label : labels)
						{
							this->cal_funcs[label](
								label,
								Aa01, Aa2, Ab01, Ab2,
								D_a, D_b, D_b_transpose,
								Ds_b01, Ds_b01_csm,
								tools);
						}
					} // end for ib2

					if( !LRI_Cal_Aux::judge_Ds_empty(Ds_result_thread) && omp_test_lock(&lock_Ds_result_add) )
					{
						LRI_Cal_Aux::add_Ds(Ds_result_thread, Ds_result);
						omp_unset_lock(&lock_Ds_result_add);
						Ds_result_thread.clear();
						Ds_result_thread.resize(Ds_result.size());
					}
				} // end for ib01
			}// end for ia2
		}// end for ia01

		if(!LRI_Cal_Aux::judge_Ds_empty(Ds_result_thread))
		{
			omp_set_lock(&lock_Ds_result_add);
			LRI_Cal_Aux::add_Ds(Ds_result_thread, Ds_result);
			omp_unset_lock(&lock_Ds_result_add);
			Ds_result_thread.clear();
			Ds_result_thread.resize(Ds_result.size());
		}
	} // end #pragma omp parallel

	omp_destroy_lock(&lock_Ds_result_add);
#ifdef __MKL
    mkl_set_num_threads(mkl_threads);
#endif
}


