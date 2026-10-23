// =================================
//  Author: Laiyuan Yang, Peize Lin
//  date: 2026.01.01
// =================================

#pragma once

#ifdef __GPU_RI

#include "LRI.h"
#include "LRI_Cal_Aux.h"
#include "../global/Array_Operator.h"
#include "../global/Map_Operator.h"
#include "../global/Tensor_Multiply.h"

#include "../global/gpu/GPU_Backend.h"
#include "gpu/GPU_Data_Input.h"
#include "gpu/GPU_Data_Mul.h"
#include "gpu/GPU_Data_Tmp.h"
#include "gpu/GPU_Data_Output.h"
#include "gpu/Dim.h"

#include <omp.h>

#ifdef __MKL_RI
#include <mkl_service.h>
#endif

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void LRI<TA,Tcell,Ndim,Tdata>::cal_loop3_GPU(
	const std::vector<Label::ab_ab> &labels,
	std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_result,
	const double fac_add_Ds)
{
	using namespace Array_Operator;

	constexpr std::size_t memory_limit = std::size_t(5)*1024*1024*1024;

	const Data_Pack_Wrapper<TA,TC,Tdata> data_wrapper(this->data_pool, this->data_ab_name);
	const LRI_Cal_Tools<TA,TC,Tdata> tools(this->period, this->data_pool, this->data_ab_name);

	std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_a_transpose, Ds_b_transpose;
	std::tie(Ds_a_transpose, Ds_b_transpose) = tools.cal_Ds_transpose(labels);

	#ifdef __MKL_RI
	const std::size_t mkl_threads = mkl_get_max_threads();
	mkl_set_num_threads(1);
	#endif

	GPU_Backend::Context gpu_context(this->mpi_comm);
	GPU_Backend::Queue &queue = gpu_context.queue();

	{
		for(const Label::ab_ab &label : labels)
		{
			const std::vector<TA>  list_Aa01_Da = LRI_Cal_Aux::filter_list_map( this->parallel->list_A[Label_Tools::to_Aab_Aab(label)].a01, data_wrapper(Label::ab::a).Ds_ab );
			const std::vector<TAC> list_Ab01_Db = LRI_Cal_Aux::filter_list_map( this->parallel->list_A[Label_Tools::to_Aab_Aab(label)].b01, data_wrapper(Label::ab::b).Ds_ab );
			const std::vector<TAC> list_Aa2_Da  = LRI_Cal_Aux::filter_list_set( this->parallel->list_A[Label_Tools::to_Aab_Aab(label)].a2,  data_wrapper(Label::ab::a).index_Ds_ab[0] );
			const std::vector<TAC> list_Ab2_Db  = LRI_Cal_Aux::filter_list_set( this->parallel->list_A[Label_Tools::to_Aab_Aab(label)].b2,  data_wrapper(Label::ab::b).index_Ds_ab[0] );
			switch(label)
			{

			  // Aab_Aab::a01b01_a01b01

				case Label::ab_ab::a0b0_a1b1:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map( LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b0).Ds_ab ),
						data_wrapper(Label::ab::a1b1).Ds_ab );
					const std::vector<TAC> &list_Aa2 =
						list_Aa2_Da;
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set( LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b0).index_Ds_ab[0]),
						data_wrapper(Label::ab::a1b1).index_Ds_ab[0]);
					const std::vector<TAC> &list_Ab2 =
						list_Ab2_Db;

					GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_a0b0, rDs_a1b1;
					GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// D_mul = D_b * D_a0b0 * D_a1b1
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01)
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const Tensor<Tdata> &D_a0b0 = tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01);
								if(D_a0b0.empty())	continue;
								const Tensor<Tdata> &D_a1b1 = tools.get_Ds_ab(Label::ab::a1b1, Aa01, Ab01);
								if(D_a1b1.empty())	continue;
								for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2)
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if (this->filter_atom->filter_for31(label, Aa01, Ab01, Ab2))	continue;
									const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
									if(D_b.empty())	continue;

									#pragma omp critical(rDs_mul)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a0b0.insert(Aa01, Ab01, D_a0b0);
										rDs_a1b1.insert(Aa01, Ab01, D_a1b1);
										const GPU_Data::Pack &D_tmp = rDs_tmp.insert({D_b.shape[1], D_b.shape[2], D_a0b0.shape[0]});
										const GPU_Data::Pack &D_mul = rDs_mul.insert_1st(Aa01, Ab2, {D_tmp.shape[1], D_tmp.shape[2], D_a1b1.shape[0]});
										dim_0.input({D_b.shape[1], D_b.shape[2]}, D_a0b0.shape[0], D_b.shape[0]);
										dim_1.input({D_tmp.shape[1], D_tmp.shape[2]}, D_a1b1.shape[0], D_a1b1.shape[1]);
									}
									rDs_b.insert_data(D_b);
									rDs_a0b0.insert_data(D_a0b0);
									rDs_a1b1.insert_data(D_a1b1);
								} // end for Aa01
							} // end for Ab2
						} // end for Ab01
					} // end omp parallel

					rDs_b.upload(queue);
					rDs_a0b0.upload(queue);
					rDs_a1b1.upload(queue);
					const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
					rDs_mul.upload_1st(queue);
					dim_0.upload(queue);
					dim_1.upload(queue);

					constexpr bool C0_left = true;
					GPU_Backend::gemmVbatched2s(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_b.d_array, rDs_a0b0.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{0}, rDs_tmp.d_array,
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a1b1.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_1,
						C0_left,
						rDs_mul.h_array_1.size(), rDs_tmp_segments_size, queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul * D_a
					GPU_Data::Input<TA, TAC, Tdata> rDs_a;
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ia2 = 0; ia2 < list_Aa2.size(); ++ia2) // F
							{
								const TAC &Aa2 = list_Aa2[ia2];
								if (this->filter_atom->filter_for2(label, Aa01, Aa2))	continue;
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if (D_a.empty())	continue;
								for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2) // G
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if (this->filter_atom->filter_for32(label, Aa01, Aa2, Ab2))	continue;
									const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab2);
									if (!D_mul.exist)	continue;

									#pragma omp critical(rDs_output)
									{
										rDs_mul.insert_2nd(D_mul);
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_output.insert( Aa2.first, {Ab2.first, (Ab2.second - Aa2.second) % period}, {D_a.shape[2], D_mul.shape[0]});
										dim_2.input(D_a.shape[2], D_mul.shape[0], {D_a.shape[0], D_a.shape[1]});
									}
									rDs_a.insert_data(D_a);
								} // end for Ab2
							} // end for Aa2
						} // end for Aa01
					} // end omp parallel

					rDs_output.upload(queue);
					rDs_mul.upload_2nd(queue);
					rDs_a.upload(queue);
					dim_2.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a.d_array, rDs_mul.d_array_2,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						rDs_output.h_array.size(), queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a0b0_a1b1

				case Label::ab_ab::a0b1_a1b0:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map( LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b1).Ds_ab ),
						data_wrapper(Label::ab::a1b0).Ds_ab );
					const std::vector<TAC> &list_Aa2 =
						list_Aa2_Da;
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set( LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b1).index_Ds_ab[0]),
						data_wrapper(Label::ab::a1b0).index_Ds_ab[0]);
					const std::vector<TAC> &list_Ab2 =
						list_Ab2_Db;

					GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_a0b1, rDs_a1b0;
					GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// D_mul = D_b * D_a0b1 * D_a1b0
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01)
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const Tensor<Tdata> &D_a0b1 = tools.get_Ds_ab(Label::ab::a0b1, Aa01, Ab01);
								if(D_a0b1.empty())	continue;
								const Tensor<Tdata> &D_a1b0 = tools.get_Ds_ab(Label::ab::a1b0, Aa01, Ab01);
								if(D_a1b0.empty())	continue;
								for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2)
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if (this->filter_atom->filter_for31(label, Aa01, Ab01, Ab2))	continue;
									const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
									if(D_b.empty())	continue;

									#pragma omp critical(rDs_mul)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a0b1.insert(Aa01, Ab01, D_a0b1);
										rDs_a1b0.insert(Aa01, Ab01, D_a1b0);
										const GPU_Data::Pack &D_tmp = rDs_tmp.insert({D_b.shape[1], D_b.shape[2], D_a1b0.shape[0]});
										const GPU_Data::Pack &D_mul = rDs_mul.insert_1st(Aa01, Ab2, {D_tmp.shape[1], D_tmp.shape[2], D_a0b1.shape[0]});
										dim_0.input({D_b.shape[1], D_b.shape[2]}, D_a1b0.shape[0], D_b.shape[0]);
										dim_1.input({D_tmp.shape[1], D_tmp.shape[2]}, D_a0b1.shape[0], D_a0b1.shape[1]);
									}
									rDs_b.insert_data(D_b);
									rDs_a0b1.insert_data(D_a0b1);
									rDs_a1b0.insert_data(D_a1b0);
								} // end for Aa01
							} // end for Ab2
						} // end for Ab01
					} // end omp parallel

					rDs_b.upload(queue);
					rDs_a0b1.upload(queue);
					rDs_a1b0.upload(queue);
					const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
					rDs_mul.upload_1st(queue);
					dim_0.upload(queue);
					dim_1.upload(queue);

					constexpr bool C0_left = true;
					GPU_Backend::gemmVbatched2s(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_b.d_array, rDs_a1b0.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{0}, rDs_tmp.d_array,
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a0b1.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_1,
						C0_left,
						rDs_mul.h_array_1.size(), rDs_tmp_segments_size, queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul * D_a
					GPU_Data::Input<TA, TAC, Tdata> rDs_a;
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ia2 = 0; ia2 < list_Aa2.size(); ++ia2) // F
							{
								const TAC &Aa2 = list_Aa2[ia2];
								if (this->filter_atom->filter_for2(label, Aa01, Aa2))	continue;
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if (D_a.empty())	continue;
								for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2) // G
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if (this->filter_atom->filter_for32(label, Aa01, Aa2, Ab2))	continue;
									const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab2);
									if (!D_mul.exist)	continue;

									#pragma omp critical(rDs_output)
									{
										rDs_mul.insert_2nd(D_mul);
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_output.insert( Aa2.first, {Ab2.first, (Ab2.second - Aa2.second) % period}, {D_a.shape[2], D_mul.shape[0]});
										dim_2.input(D_a.shape[2], D_mul.shape[0], {D_a.shape[0], D_a.shape[1]});
									}
									rDs_a.insert_data(D_a);
								} // end for Ab2
							} // end for Aa2
						} // end for Aa01
					} // end omp parallel

					rDs_output.upload(queue);
					rDs_mul.upload_2nd(queue);
					rDs_a.upload(queue);
					dim_2.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a.d_array, rDs_mul.d_array_2,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						rDs_output.h_array.size(), queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a0b1_a1b0

			  // Aab_Aab::a01b01_a01b2

				case Label::ab_ab::a0b0_a1b2:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map( LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b0).Ds_ab ),
						data_wrapper(Label::ab::a1b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 =
						list_Aa2_Da;
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b0).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a1b2).index_Ds_ab[0]);

					GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_a1b2;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
					GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// b0b1a1 = b0b1b2 * a1b2
					// D_mul = D_b * D_a1b2
					#pragma omp parallel
					{
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for1(label,Ab01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
							{
								const TAC &Ab2 = list_Ab2[ib2];
								if(this->filter_atom->filter_for2(label,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								for (std::size_t ia01= 0; ia01<list_Aa01.size(); ++ia01)
								{
									const TA &Aa01 = list_Aa01[ia01];
									if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
									const Tensor<Tdata> &D_a1b2 = tools.get_Ds_ab(Label::ab::a1b2, Aa01, Ab2);
									if(D_a1b2.empty())	continue;

									#pragma omp critical(rDs_mul)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a1b2.insert(Aa01, Ab2, D_a1b2);
										const GPU_Data::Pack &D_mul = rDs_mul.insert_1st(Aa01, Ab01, {D_b.shape[0], D_b.shape[1], D_a1b2.shape[0]});
										dim_0.input({D_b.shape[0], D_b.shape[1]}, D_a1b2.shape[0], D_b.shape[2]);
									}
									rDs_b.insert_data(D_b);
									rDs_a1b2.insert_data(D_a1b2);
								} // end for Ab2
							} // end for Aa01
						} // end for Ab01
					} // end omp parallel

					rDs_b.upload(queue);
					rDs_a1b2.upload(queue);
					rDs_mul.upload_1st(queue);

					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::NoTrans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_b.d_array, rDs_a1b2.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_1,
						rDs_mul.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul * D_a * D_a0b0
					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_a0b0;

					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01) // G
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const Tensor<Tdata> &D_a0b0 = tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01);
								if(D_a0b0.empty())	continue;
								const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab01);
								if (!D_mul.exist)	continue;
								for (std::size_t ia2 = 0; ia2 < list_Aa2.size(); ++ia2) // F
								{
									const TAC &Aa2 = list_Aa2[ia2];
									if (this->filter_atom->filter_for32(label, Aa01, Aa2, Ab01))	continue;
									const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
									if (D_a.empty())	continue;

									#pragma omp critical(rDs_output)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a0b0.insert(Aa01, Ab01, D_a0b0);
										rDs_mul.insert_2nd(D_mul);
										const GPU_Data::Pack &D_tmp = rDs_tmp.insert({D_mul.shape[1], D_mul.shape[2], D_a0b0.shape[0]});
										rDs_output.insert(Aa2.first, {Ab01.first, (Ab01.second - Aa2.second) % period}, {D_a.shape[2], D_tmp.shape[0]});
										dim_1.input({D_mul.shape[1], D_mul.shape[2]}, D_a0b0.shape[0], D_a0b0.shape[1]);
										dim_2.input(D_a.shape[2], D_tmp.shape[0], {D_a.shape[0], D_a.shape[1]});
									}
									rDs_a.insert_data(D_a);
									rDs_a0b0.insert_data(D_a0b0);
								} // end for Ab01
							}  // end for Aa2
						} // end for Aa01
					} // end omp parallel

					const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
					rDs_output.upload(queue);
					rDs_mul.upload_2nd(queue);
					rDs_a.upload(queue);
					rDs_a0b0.upload(queue);

					dim_1.upload(queue);
					dim_2.upload(queue);

					constexpr bool C0_left = false;
					GPU_Backend::gemmVbatched2s(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_2, rDs_a0b0.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{0}, rDs_tmp.d_array,
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						C0_left,
						rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a0b0_a1b2

				case Label::ab_ab::a0b1_a1b2:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map( LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b1).Ds_ab ),
						data_wrapper(Label::ab::a1b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 =
						list_Aa2_Da;
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b1).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a1b2).index_Ds_ab[0]);

					GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_a1b2;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
					GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// a1b0b1 = a1b2 * b0b1b2
					// D_mul = D_b * D_a1b2
					#pragma omp parallel
					{
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for1(label,Ab01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
							{
								const TAC &Ab2 = list_Ab2[ib2];
								if(this->filter_atom->filter_for2(label,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								for (std::size_t ia01= 0; ia01<list_Aa01.size(); ++ia01)
								{
									const TA &Aa01 = list_Aa01[ia01];
									if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
									const Tensor<Tdata> &D_a1b2 = tools.get_Ds_ab(Label::ab::a1b2, Aa01, Ab2);
									if(D_a1b2.empty())	continue;

									#pragma omp critical(rDs_mul)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a1b2.insert(Aa01, Ab2, D_a1b2);
										const GPU_Data::Pack &D_mul = rDs_mul.insert_1st(Aa01, Ab01, {D_a1b2.shape[0], D_b.shape[0], D_b.shape[1]});
										dim_0.input(D_a1b2.shape[0], {D_b.shape[0], D_b.shape[1]}, D_a1b2.shape[1]);
									}
									rDs_b.insert_data(D_b);
									rDs_a1b2.insert_data(D_a1b2);
								} // end for Ab2
							} // end for Aa01
						} // end for Ab01
					} // end omp parallel

					rDs_b.upload(queue);
					rDs_a1b2.upload(queue);
					rDs_mul.upload_1st(queue);

					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::NoTrans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a1b2.d_array, rDs_b.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_1,
						rDs_mul.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul * D_a * D_a0b1
					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_a0b1;

					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01) // G
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const Tensor<Tdata> &D_a0b1 = tools.get_Ds_ab(Label::ab::a0b1, Aa01, Ab01);
								if(D_a0b1.empty())	continue;
								const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab01);
								if (!D_mul.exist)	continue;
								for (std::size_t ia2 = 0; ia2 < list_Aa2.size(); ++ia2) // F
								{
									const TAC &Aa2 = list_Aa2[ia2];
									if (this->filter_atom->filter_for32(label, Aa01, Aa2, Ab01))	continue;
									const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
									if (D_a.empty())	continue;

									#pragma omp critical(rDs_output)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a0b1.insert(Aa01, Ab01, D_a0b1);
										rDs_mul.insert_2nd(D_mul);
										const GPU_Data::Pack &D_tmp = rDs_tmp.insert({D_a0b1.shape[0], D_mul.shape[0], D_mul.shape[1]});
										rDs_output.insert(Aa2.first, {Ab01.first, (Ab01.second - Aa2.second) % period}, {D_a.shape[2], D_tmp.shape[2]});
										dim_1.input(D_a0b1.shape[0], {D_mul.shape[0], D_mul.shape[1]}, D_a0b1.shape[1]);
										dim_2.input(D_a.shape[2], D_tmp.shape[2], {D_a.shape[0], D_a.shape[1]});
									}
									rDs_a.insert_data(D_a);
									rDs_a0b1.insert_data(D_a0b1);
								} // end for Ab01
							}  // end for Aa2
						} // end for Aa01
					} // end omp parallel

					const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
					rDs_output.upload(queue);
					rDs_mul.upload_2nd(queue);
					rDs_a.upload(queue);
					rDs_a0b1.upload(queue);

					dim_1.upload(queue);
					dim_2.upload(queue);

					constexpr bool C0_left = false;
					GPU_Backend::gemmVbatched2s(
						GPU_Backend::NoTrans, GPU_Backend::Trans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a0b1.d_array, rDs_mul.d_array_2,
						GPU_Backend::Type_to_GPU<Tdata>{0}, rDs_tmp.d_array,
						GPU_Backend::Trans, GPU_Backend::NoTrans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						C0_left,
						rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a0b1_a1b2

				case Label::ab_ab::a0b2_a1b0:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map( LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b0).Ds_ab ),
						data_wrapper(Label::ab::a0b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 =
						list_Aa2_Da;
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a1b0).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a0b2).index_Ds_ab[0]);

					GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_a0b2;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
					GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// b0b1a0 = b0b1b2 * a0b2
					// D_mul = D_b * D_a0b2
					#pragma omp parallel
					{
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for1(label,Ab01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
							{
								const TAC &Ab2 = list_Ab2[ib2];
								if(this->filter_atom->filter_for2(label,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								for (std::size_t ia01= 0; ia01<list_Aa01.size(); ++ia01)
								{
									const TA &Aa01 = list_Aa01[ia01];
									if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
									const Tensor<Tdata> &D_a0b2 = tools.get_Ds_ab(Label::ab::a0b2, Aa01, Ab2);
									if(D_a0b2.empty())	continue;

									#pragma omp critical(rDs_mul)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a0b2.insert(Aa01, Ab2, D_a0b2);
										const GPU_Data::Pack &D_mul = rDs_mul.insert_1st(Aa01, Ab01, {D_b.shape[0], D_b.shape[1], D_a0b2.shape[0]});
										dim_0.input({D_b.shape[0], D_b.shape[1]}, D_a0b2.shape[0], D_b.shape[2]);
									}
									rDs_b.insert_data(D_b);
									rDs_a0b2.insert_data(D_a0b2);
								} // end for Ab2
							} // end for Aa01
						} // end for Ab01
					} // end omp parallel

					rDs_b.upload(queue);
					rDs_a0b2.upload(queue);
					rDs_mul.upload_1st(queue);

					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::NoTrans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_b.d_array, rDs_a0b2.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_1,
						rDs_mul.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul * D_a * D_a1b0
					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_a1b0;

					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01) // G
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const Tensor<Tdata> &D_a1b0 = tools.get_Ds_ab(Label::ab::a1b0, Aa01, Ab01);
								if(D_a1b0.empty())	continue;
								const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab01);
								if (!D_mul.exist)	continue;
								for (std::size_t ia2 = 0; ia2 < list_Aa2.size(); ++ia2) // F
								{
									const TAC &Aa2 = list_Aa2[ia2];
									if (this->filter_atom->filter_for32(label, Aa01, Aa2, Ab01))	continue;
									const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
									if (D_a.empty())	continue;

									#pragma omp critical(rDs_output)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a1b0.insert(Aa01, Ab01, D_a1b0);
										rDs_mul.insert_2nd(D_mul);
										const GPU_Data::Pack &D_tmp = rDs_tmp.insert({D_mul.shape[1], D_mul.shape[2], D_a1b0.shape[0]});
										rDs_output.insert(Aa2.first, {Ab01.first, (Ab01.second - Aa2.second) % period}, {D_a.shape[2], D_tmp.shape[0]});
										dim_1.input({D_mul.shape[1], D_mul.shape[2]}, D_a1b0.shape[0], D_a1b0.shape[1]);
										dim_2.input(D_a.shape[2], D_tmp.shape[0], {D_a.shape[0], D_a.shape[1]});
									}
									rDs_a.insert_data(D_a);
									rDs_a1b0.insert_data(D_a1b0);
								} // end for Ab01
							}  // end for Aa2
						} // end for Aa01
					} // end omp parallel

					const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
					rDs_output.upload(queue);
					rDs_mul.upload_2nd(queue);
					rDs_a.upload(queue);
					rDs_a1b0.upload(queue);

					dim_1.upload(queue);
					dim_2.upload(queue);

					constexpr bool C0_left = false;
					GPU_Backend::gemmVbatched2s(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_2, rDs_a1b0.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{0}, rDs_tmp.d_array,
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						C0_left,
						rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a0b2_a1b0

				case Label::ab_ab::a0b2_a1b1:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map( LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b1).Ds_ab ),
						data_wrapper(Label::ab::a0b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 =
						list_Aa2_Da;
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a1b1).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a0b2).index_Ds_ab[0]);

					GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_a0b2;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
					GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// a0b0b1 = a0b2 * b0b1b2
					// D_mul = D_b * D_a0b2
					#pragma omp parallel
					{
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for1(label,Ab01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
							{
								const TAC &Ab2 = list_Ab2[ib2];
								if(this->filter_atom->filter_for2(label,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								for (std::size_t ia01= 0; ia01<list_Aa01.size(); ++ia01)
								{
									const TA &Aa01 = list_Aa01[ia01];
									if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
									const Tensor<Tdata> &D_a0b2 = tools.get_Ds_ab(Label::ab::a0b2, Aa01, Ab2);
									if(D_a0b2.empty())	continue;

									#pragma omp critical(rDs_mul)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a0b2.insert(Aa01, Ab2, D_a0b2);
										const GPU_Data::Pack &D_mul = rDs_mul.insert_1st(Aa01, Ab01, {D_a0b2.shape[0], D_b.shape[0], D_b.shape[1]});
										dim_0.input(D_a0b2.shape[0], {D_b.shape[0], D_b.shape[1]}, D_a0b2.shape[1]);
									}
									rDs_b.insert_data(D_b);
									rDs_a0b2.insert_data(D_a0b2);
								} // end for Ab2
							} // end for Aa01
						} // end for Ab01
					} // end omp parallel

					rDs_b.upload(queue);
					rDs_a0b2.upload(queue);
					rDs_mul.upload_1st(queue);

					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::NoTrans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a0b2.d_array, rDs_b.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_1,
						rDs_mul.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul * D_a * D_a1b1
					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_a1b1;

					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01) // G
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const Tensor<Tdata> &D_a1b1 = tools.get_Ds_ab(Label::ab::a1b1, Aa01, Ab01);
								if(D_a1b1.empty())	continue;
								const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab01);
								if (!D_mul.exist)	continue;
								for (std::size_t ia2 = 0; ia2 < list_Aa2.size(); ++ia2) // F
								{
									const TAC &Aa2 = list_Aa2[ia2];
									if (this->filter_atom->filter_for32(label, Aa01, Aa2, Ab01))	continue;
									const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
									if (D_a.empty())	continue;

									#pragma omp critical(rDs_output)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a1b1.insert(Aa01, Ab01, D_a1b1);
										rDs_mul.insert_2nd(D_mul);
										const GPU_Data::Pack &D_tmp = rDs_tmp.insert({D_a1b1.shape[0], D_mul.shape[0], D_mul.shape[1]});
										rDs_output.insert(Aa2.first, {Ab01.first, (Ab01.second - Aa2.second) % period}, {D_a.shape[2], D_tmp.shape[2]});
										dim_1.input(D_a1b1.shape[0], {D_mul.shape[0], D_mul.shape[1]}, D_a1b1.shape[1]);
										dim_2.input(D_a.shape[2], D_tmp.shape[2], {D_a.shape[0], D_a.shape[1]});
									}
									rDs_a.insert_data(D_a);
									rDs_a1b1.insert_data(D_a1b1);
								} // end for Ab01
							}  // end for Aa2
						} // end for Aa01
					} // end omp parallel

					const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
					rDs_output.upload(queue);
					rDs_mul.upload_2nd(queue);
					rDs_a.upload(queue);
					rDs_a1b1.upload(queue);

					dim_1.upload(queue);
					dim_2.upload(queue);

					constexpr bool C0_left = false;
					GPU_Backend::gemmVbatched2s(
						GPU_Backend::NoTrans, GPU_Backend::Trans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a1b1.d_array, rDs_mul.d_array_2,
						GPU_Backend::Type_to_GPU<Tdata>{0}, rDs_tmp.d_array,
						GPU_Backend::Trans, GPU_Backend::NoTrans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						C0_left,
						rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a0b2_a1b1

			  // Aab_Aab::a01b01_a2b01

				case Label::ab_ab::a0b0_a2b1:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b0).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b1).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set( LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b0).index_Ds_ab[0]),
						data_wrapper(Label::ab::a2b1).index_Ds_ab[0]);
					const std::vector<TAC> &list_Ab2 =
						list_Ab2_Db;

					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_a2b1;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
					GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// b1a1a0 = a2b1 * a1a0a2
					// D_mul = D_a * D_a2b1
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for1(label,Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ia2=0; ia2<list_Aa2.size(); ++ia2)
							{
								const TAC &Aa2 = list_Aa2[ia2];
								if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								for (std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
								{
									const TAC &Ab01 = list_Ab01[ib01];
									if(this->filter_atom->filter_for31(label,Aa01,Ab01,Aa2))	continue;
									const Tensor<Tdata> &D_a2b1 = tools.get_Ds_ab(Label::ab::a2b1, Aa2, Ab01);
									if(D_a2b1.empty())	continue;

									#pragma omp critical(rDs_mul)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a2b1.insert(Aa2.first, {Ab01.first, (Ab01.second - Aa2.second) % this->period}, D_a2b1);
										const GPU_Data::Pack &D_mul = rDs_mul.insert_1st(Aa01, Ab01, {D_a2b1.shape[1], D_a.shape[0], D_a.shape[1]});
										dim_0.input(D_a2b1.shape[1], {D_a.shape[0], D_a.shape[1]}, D_a2b1.shape[0]);
									}
									rDs_a.insert_data(D_a);
									rDs_a2b1.insert_data(D_a2b1);

								} // end for Aa2
							} // end for Ab01
						} // end for Aa01
					} // end omp parallel

					rDs_a.upload(queue);
					rDs_a2b1.upload(queue);
					rDs_mul.upload_1st(queue);
					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a2b1.d_array, rDs_a.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_1,
						rDs_mul.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul * D_a0b0 * D_b
					GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_a0b0;

					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01) // G
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const Tensor<Tdata> &D_a0b0 = tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01);
								if(D_a0b0.empty())	continue;
								const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab01);
								if (!D_mul.exist)	continue;
								for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2)
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if (this->filter_atom->filter_for32(label, Aa01, Ab01, Ab2))	continue;
									const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
									if(D_b.empty())	continue;

									#pragma omp critical(rDs_output)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a0b0.insert(Aa01, Ab01, D_a0b0);
										rDs_mul.insert_2nd(D_mul);
										const GPU_Data::Pack &D_tmp = rDs_tmp.insert({D_a0b0.shape[1], D_mul.shape[0], D_mul.shape[1]});
										rDs_output.insert(Aa01, Ab2, {D_tmp.shape[2], D_b.shape[2]});
										dim_1.input(D_a0b0.shape[1], {D_mul.shape[0], D_mul.shape[1]}, D_a0b0.shape[0]);
										dim_2.input(D_tmp.shape[2], D_b.shape[2], {D_b.shape[0], D_b.shape[1]});
									}
									rDs_b.insert_data(D_b);
									rDs_a0b0.insert_data(D_a0b0);
								} // end for Aa01
							} // end for Ab2
						} // end for Ab01
					} // end omp parallel

					const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
					rDs_output.upload(queue);
					rDs_mul.upload_2nd(queue);
					rDs_b.upload(queue);
					rDs_a0b0.upload(queue);

					dim_1.upload(queue);
					dim_2.upload(queue);

					constexpr bool C0_left = true;
					GPU_Backend::gemmVbatched2s(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a0b0.d_array, rDs_mul.d_array_2,
						GPU_Backend::Type_to_GPU<Tdata>{0}, rDs_tmp.d_array,
						GPU_Backend::Trans, GPU_Backend::NoTrans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_b.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						C0_left,
						rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a0b0_a2b1

				case Label::ab_ab::a0b1_a2b0:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b1).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b0).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set( LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b1).index_Ds_ab[0]),
						data_wrapper(Label::ab::a2b0).index_Ds_ab[0]);
					const std::vector<TAC> &list_Ab2 =
						list_Ab2_Db;

					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_a2b0;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
					GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// a0a1b0 = a0a1a2 * a2b0
					// D_mul = D_a * D_a2b0
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for1(label,Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ia2=0; ia2<list_Aa2.size(); ++ia2)
							{
								const TAC &Aa2 = list_Aa2[ia2];
								if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								for (std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
								{
									const TAC &Ab01 = list_Ab01[ib01];
									if(this->filter_atom->filter_for31(label,Aa01,Ab01,Aa2))	continue;
									const Tensor<Tdata> &D_a2b0 = tools.get_Ds_ab(Label::ab::a2b0, Aa2, Ab01);
									if(D_a2b0.empty())	continue;

									#pragma omp critical(rDs_mul)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a2b0.insert(Aa2.first, {Ab01.first, (Ab01.second - Aa2.second) % this->period}, D_a2b0);
										const GPU_Data::Pack &D_mul = rDs_mul.insert_1st(Aa01, Ab01, {D_a.shape[0], D_a.shape[1], D_a2b0.shape[1]});
										dim_0.input({D_a.shape[0], D_a.shape[1]}, D_a2b0.shape[1], D_a.shape[2]);
									}
									rDs_a.insert_data(D_a);
									rDs_a2b0.insert_data(D_a2b0);

								} // end for Aa2
							} // end for Ab01
						} // end for Aa01
					} // end omp parallel

					rDs_a.upload(queue);
					rDs_a2b0.upload(queue);
					rDs_mul.upload_1st(queue);
					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::NoTrans, GPU_Backend::NoTrans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a.d_array, rDs_a2b0.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_1,
						rDs_mul.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul * D_a0b1 * D_b
					GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_a0b1;

					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01) // G
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const Tensor<Tdata> &D_a0b1 = tools.get_Ds_ab(Label::ab::a0b1, Aa01, Ab01);
								if(D_a0b1.empty())	continue;
								const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab01);
								if (!D_mul.exist)	continue;
								for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2)
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if (this->filter_atom->filter_for32(label, Aa01, Ab01, Ab2))	continue;
									const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
									if(D_b.empty())	continue;

									#pragma omp critical(rDs_output)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a0b1.insert(Aa01, Ab01, D_a0b1);
										rDs_mul.insert_2nd(D_mul);
										const GPU_Data::Pack &D_tmp = rDs_tmp.insert({D_mul.shape[1], D_mul.shape[2], D_a0b1.shape[1]});
										rDs_output.insert(Aa01, Ab2, {D_tmp.shape[0], D_b.shape[2]});
										dim_1.input({D_mul.shape[1], D_mul.shape[2]}, D_a0b1.shape[1], D_a0b1.shape[0]);
										dim_2.input(D_tmp.shape[0], D_b.shape[2], {D_b.shape[0], D_b.shape[1]});
									}
									rDs_b.insert_data(D_b);
									rDs_a0b1.insert_data(D_a0b1);
								} // end for Aa01
							} // end for Ab2
						} // end for Ab01
					} // end omp parallel

					const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
					rDs_output.upload(queue);
					rDs_mul.upload_2nd(queue);
					rDs_b.upload(queue);
					rDs_a0b1.upload(queue);

					dim_1.upload(queue);
					dim_2.upload(queue);

					constexpr bool C0_left = true;
					GPU_Backend::gemmVbatched2s(
						GPU_Backend::Trans, GPU_Backend::NoTrans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_2, rDs_a0b1.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{0}, rDs_tmp.d_array,
						GPU_Backend::NoTrans, GPU_Backend::NoTrans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_b.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						C0_left,
						rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a0b1_a2b0

				case Label::ab_ab::a1b0_a2b1:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b0).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b1).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set( LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a1b0).index_Ds_ab[0]),
						data_wrapper(Label::ab::a2b1).index_Ds_ab[0]);
					const std::vector<TAC> &list_Ab2 =
						list_Ab2_Db;

					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_a2b1;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
					GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// b1a0a1 = a2b1 * a0a1a2
					// D_mul = D_a * D_a2b1
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for1(label,Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ia2=0; ia2<list_Aa2.size(); ++ia2)
							{
								const TAC &Aa2 = list_Aa2[ia2];
								if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								for (std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
								{
									const TAC &Ab01 = list_Ab01[ib01];
									if(this->filter_atom->filter_for31(label,Aa01,Ab01,Aa2))	continue;
									const Tensor<Tdata> &D_a2b1 = tools.get_Ds_ab(Label::ab::a2b1, Aa2, Ab01);
									if(D_a2b1.empty())	continue;

									#pragma omp critical(rDs_mul)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a2b1.insert(Aa2.first, {Ab01.first, (Ab01.second - Aa2.second) % this->period}, D_a2b1);
										const GPU_Data::Pack &D_mul = rDs_mul.insert_1st(Aa01, Ab01, {D_a2b1.shape[1], D_a.shape[0], D_a.shape[1]});
										dim_0.input(D_a2b1.shape[1], {D_a.shape[0], D_a.shape[1]}, D_a2b1.shape[0]);
									}
									rDs_a.insert_data(D_a);
									rDs_a2b1.insert_data(D_a2b1);

								} // end for Aa2
							} // end for Ab01
						} // end for Aa01
					} // end omp parallel

					rDs_a.upload(queue);
					rDs_a2b1.upload(queue);
					rDs_mul.upload_1st(queue);
					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a2b1.d_array, rDs_a.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_1,
						rDs_mul.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul * D_a1b0 * D_b
					GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_a1b0;

					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01) // G
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const Tensor<Tdata> &D_a1b0 = tools.get_Ds_ab(Label::ab::a1b0, Aa01, Ab01);
								if(D_a1b0.empty())	continue;
								const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab01);
								if (!D_mul.exist)	continue;
								for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2)
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if (this->filter_atom->filter_for32(label, Aa01, Ab01, Ab2))	continue;
									const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
									if(D_b.empty())	continue;

									#pragma omp critical(rDs_output)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a1b0.insert(Aa01, Ab01, D_a1b0);
										rDs_mul.insert_2nd(D_mul);
										const GPU_Data::Pack &D_tmp = rDs_tmp.insert({D_a1b0.shape[1], D_mul.shape[0], D_mul.shape[1]});
										rDs_output.insert(Aa01, Ab2, {D_tmp.shape[2], D_b.shape[2]});
										dim_1.input(D_a1b0.shape[1], {D_mul.shape[0], D_mul.shape[1]}, D_a1b0.shape[0]);
										dim_2.input(D_tmp.shape[2], D_b.shape[2], {D_b.shape[0], D_b.shape[1]});
									}
									rDs_b.insert_data(D_b);
									rDs_a1b0.insert_data(D_a1b0);
								} // end for Aa01
							} // end for Ab2
						} // end for Ab01
					} // end omp parallel

					const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
					rDs_output.upload(queue);
					rDs_mul.upload_2nd(queue);
					rDs_b.upload(queue);
					rDs_a1b0.upload(queue);

					dim_1.upload(queue);
					dim_2.upload(queue);

					constexpr bool C0_left = true;
					GPU_Backend::gemmVbatched2s(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a1b0.d_array, rDs_mul.d_array_2,
						GPU_Backend::Type_to_GPU<Tdata>{0}, rDs_tmp.d_array,
						GPU_Backend::Trans, GPU_Backend::NoTrans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_b.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						C0_left,
						rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a1b0_a2b1

				case Label::ab_ab::a1b1_a2b0:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b1).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b0).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set( LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a1b1).index_Ds_ab[0]),
						data_wrapper(Label::ab::a2b0).index_Ds_ab[0]);
					const std::vector<TAC> &list_Ab2 =
						list_Ab2_Db;

					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_a2b0;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
					GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// a1a0b0 = a1a0a2 * a2b0
					// D_mul = D_a * D_a2b0
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for1(label,Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ia2=0; ia2<list_Aa2.size(); ++ia2)
							{
								const TAC &Aa2 = list_Aa2[ia2];
								if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								for (std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
								{
									const TAC &Ab01 = list_Ab01[ib01];
									if(this->filter_atom->filter_for31(label,Aa01,Ab01,Aa2))	continue;
									const Tensor<Tdata> &D_a2b0 = tools.get_Ds_ab(Label::ab::a2b0, Aa2, Ab01);
									if(D_a2b0.empty())	continue;

									#pragma omp critical(rDs_mul)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a2b0.insert(Aa2.first, {Ab01.first, (Ab01.second - Aa2.second) % this->period}, D_a2b0);
										const GPU_Data::Pack &D_mul = rDs_mul.insert_1st(Aa01, Ab01, {D_a.shape[0], D_a.shape[1], D_a2b0.shape[1]});
										dim_0.input({D_a.shape[0], D_a.shape[1]}, D_a2b0.shape[1], D_a.shape[2]);
									}
									rDs_a.insert_data(D_a);
									rDs_a2b0.insert_data(D_a2b0);

								} // end for Aa2
							} // end for Ab01
						} // end for Aa01
					} // end omp parallel

					rDs_a.upload(queue);
					rDs_a2b0.upload(queue);
					rDs_mul.upload_1st(queue);
					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::NoTrans, GPU_Backend::NoTrans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a.d_array, rDs_a2b0.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_1,
						rDs_mul.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul * D_a1b1 * D_b
					GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_a1b1;

					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01) // G
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const Tensor<Tdata> &D_a1b1 = tools.get_Ds_ab(Label::ab::a1b1, Aa01, Ab01);
								if(D_a1b1.empty())	continue;
								const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab01);
								if (!D_mul.exist)	continue;
								for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2)
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if (this->filter_atom->filter_for32(label, Aa01, Ab01, Ab2))	continue;
									const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
									if(D_b.empty())	continue;

									#pragma omp critical(rDs_output)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a1b1.insert(Aa01, Ab01, D_a1b1);
										rDs_mul.insert_2nd(D_mul);
										const GPU_Data::Pack &D_tmp = rDs_tmp.insert({D_mul.shape[1], D_mul.shape[2], D_a1b1.shape[1]});
										rDs_output.insert(Aa01, Ab2, {D_tmp.shape[0], D_b.shape[2]});
										dim_1.input({D_mul.shape[1], D_mul.shape[2]}, D_a1b1.shape[1], D_a1b1.shape[0]);
										dim_2.input(D_tmp.shape[0], D_b.shape[2], {D_b.shape[0], D_b.shape[1]});
									}
									rDs_b.insert_data(D_b);
									rDs_a1b1.insert_data(D_a1b1);
								} // end for Aa01
							} // end for Ab2
						} // end for Ab01
					} // end omp parallel

					const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
					rDs_output.upload(queue);
					rDs_mul.upload_2nd(queue);
					rDs_b.upload(queue);
					rDs_a1b1.upload(queue);

					dim_1.upload(queue);
					dim_2.upload(queue);

					constexpr bool C0_left = true;
					GPU_Backend::gemmVbatched2s(
						GPU_Backend::Trans, GPU_Backend::NoTrans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_2, rDs_a1b1.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{0}, rDs_tmp.d_array,
						GPU_Backend::NoTrans, GPU_Backend::NoTrans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_b.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						C0_left,
						rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a1b1_a2b0

			  // Aab_Aab::a01b01_a2b2

				case Label::ab_ab::a0b0_a2b2:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b0).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b2).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b0).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a2b2).index_Ds_ab[0]);

					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_a2b2;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
					GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// b2a1a0 = a2b2 * a1a0a2
					// D_mul = D_a * D_a2b2
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for1(label,Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ia2 = 0; ia2 < list_Aa2.size(); ++ia2)
							{
								const TAC &Aa2 = list_Aa2[ia2];
								if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if(this->filter_atom->filter_for31(label,Aa01,Aa2,Ab2))	continue;
									const Tensor<Tdata> &D_a2b2 = tools.get_Ds_ab(Label::ab::a2b2, Aa2, Ab2);
									if(D_a2b2.empty())	continue;

									#pragma omp critical(rDs_mul)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a2b2.insert(Aa2.first, {Ab2.first, (Ab2.second - Aa2.second) % period}, D_a2b2);
										rDs_mul.insert_1st(Aa01, Ab2, {D_a2b2.shape[1], D_a.shape[0], D_a.shape[1]});
										dim_0.input(D_a2b2.shape[1], {D_a.shape[0], D_a.shape[1]}, D_a2b2.shape[0]);
									}
									rDs_a.insert_data(D_a);
									rDs_a2b2.insert_data(D_a2b2);
								} // end for Ab2
							} // end for Aa2
						} // end for Aa01
					} // end omp parallel

					rDs_a.upload(queue);
					rDs_a2b2.upload(queue);
					rDs_mul.upload_1st(queue);
					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a2b2.d_array, rDs_a.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_1,
						rDs_mul.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul * D_a0b0 * D_b
					GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_a0b0;

					#pragma omp parallel
					{
						for (std::size_t ia01 = 0; ia01 < list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01)
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const Tensor<Tdata> &D_a0b0 = tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01);
								if(D_a0b0.empty())	continue;
								for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2)
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if (this->filter_atom->filter_for32(label, Aa01, Ab01, Ab2))	continue;
									const Tensor<Tdata> &D_b = Global_Func::find(Ds_b_transpose, Ab01.first, TAC{Ab2.first, (Ab2.second-Ab01.second)%this->period});
									if(D_b.empty())	continue;
									const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab2);
									if (!D_mul.exist)	continue;

									#pragma omp critical(rDs_output)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a0b0.insert(Aa01, Ab01, D_a0b0);
										rDs_mul.insert_2nd(D_mul);
										const GPU_Data::Pack &D_tmp = rDs_tmp.insert({D_a0b0.shape[1], D_mul.shape[0], D_mul.shape[1]});
										rDs_output.insert(Aa01, Ab01, {D_tmp.shape[2], D_b.shape[0]});
										dim_1.input(D_a0b0.shape[1], {D_mul.shape[0], D_mul.shape[1]}, D_a0b0.shape[0]);
										dim_2.input(D_tmp.shape[2], D_b.shape[0], {D_b.shape[1], D_b.shape[2]});
									}
									rDs_b.insert_data(D_b);
									rDs_a0b0.insert_data(D_a0b0);
								} // end for Aa01
							} // end for Ab2
						} // end for Ab01
					} // end omp parallel

					rDs_b.upload(queue);
					rDs_a0b0.upload(queue);
					rDs_mul.upload_2nd(queue);
					const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
					rDs_output.upload(queue);

					dim_1.upload(queue);
					dim_2.upload(queue);

					constexpr bool C0_left = true;
					GPU_Backend::gemmVbatched2s(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a0b0.d_array, rDs_mul.d_array_2,
						GPU_Backend::Type_to_GPU<Tdata>{0}, rDs_tmp.d_array,
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_b.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						C0_left,
						rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a0b0_a2b2

				case Label::ab_ab::a0b1_a2b2:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b1).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b2).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b1).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a2b2).index_Ds_ab[0]);

					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_a2b2;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
					GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// b2a1a0 = a2b2 * a1a0a2
					// D_mul = D_a * D_a2b2
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for1(label,Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ia2 = 0; ia2 < list_Aa2.size(); ++ia2)
							{
								const TAC &Aa2 = list_Aa2[ia2];
								if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if(this->filter_atom->filter_for31(label,Aa01,Aa2,Ab2))	continue;
									const Tensor<Tdata> &D_a2b2 = tools.get_Ds_ab(Label::ab::a2b2, Aa2, Ab2);
									if(D_a2b2.empty())	continue;

									#pragma omp critical(rDs_mul)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a2b2.insert(Aa2.first, {Ab2.first, (Ab2.second - Aa2.second) % period}, D_a2b2);
										rDs_mul.insert_1st(Aa01, Ab2, {D_a2b2.shape[1], D_a.shape[0], D_a.shape[1]});
										dim_0.input(D_a2b2.shape[1], {D_a.shape[0], D_a.shape[1]}, D_a2b2.shape[0]);
									}
									rDs_a.insert_data(D_a);
									rDs_a2b2.insert_data(D_a2b2);
								} // end for Ab2
							} // end for Aa2
						} // end for Aa01
					} // end omp parallel

					rDs_a.upload(queue);
					rDs_a2b2.upload(queue);
					rDs_mul.upload_1st(queue);
					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a2b2.d_array, rDs_a.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_1,
						rDs_mul.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul * D_a0b1 * D_b
					GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_a0b1;

					#pragma omp parallel
					{
						for (std::size_t ia01 = 0; ia01 < list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01)
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const Tensor<Tdata> &D_a0b1 = tools.get_Ds_ab(Label::ab::a0b1, Aa01, Ab01);
								if(D_a0b1.empty())	continue;
								for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2)
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if (this->filter_atom->filter_for32(label, Aa01, Ab01, Ab2))	continue;
									const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
									if(D_b.empty())	continue;
									const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab2);
									if (!D_mul.exist)	continue;

									#pragma omp critical(rDs_output)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a0b1.insert(Aa01, Ab01, D_a0b1);
										rDs_mul.insert_2nd(D_mul);
										const GPU_Data::Pack &D_tmp = rDs_tmp.insert({D_a0b1.shape[1], D_mul.shape[0], D_mul.shape[1]});
										rDs_output.insert(Aa01, Ab01, {D_tmp.shape[2], D_b.shape[0]});
										dim_1.input(D_a0b1.shape[1], {D_mul.shape[0], D_mul.shape[1]}, D_a0b1.shape[0]);
										dim_2.input(D_tmp.shape[2], D_b.shape[0], {D_b.shape[1], D_b.shape[2]});
									}
									rDs_b.insert_data(D_b);
									rDs_a0b1.insert_data(D_a0b1);
								} // end for Aa01
							} // end for Ab2
						} // end for Ab01
					} // end omp parallel

					rDs_b.upload(queue);
					rDs_a0b1.upload(queue);
					rDs_mul.upload_2nd(queue);
					const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
					rDs_output.upload(queue);

					dim_1.upload(queue);
					dim_2.upload(queue);

					constexpr bool C0_left = true;
					GPU_Backend::gemmVbatched2s(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a0b1.d_array, rDs_mul.d_array_2,
						GPU_Backend::Type_to_GPU<Tdata>{0}, rDs_tmp.d_array,
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_b.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						C0_left,
						rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a0b1_a2b2

				case Label::ab_ab::a1b0_a2b2:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b0).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b2).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a1b0).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a2b2).index_Ds_ab[0]);

					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_a2b2;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
					GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// b2a0a1 = a2b2 * a0a1a2
					// D_mul = D_a * D_a2b2
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for1(label,Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ia2 = 0; ia2 < list_Aa2.size(); ++ia2)
							{
								const TAC &Aa2 = list_Aa2[ia2];
								if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if(this->filter_atom->filter_for31(label,Aa01,Aa2,Ab2))	continue;
									const Tensor<Tdata> &D_a2b2 = tools.get_Ds_ab(Label::ab::a2b2, Aa2, Ab2);
									if(D_a2b2.empty())	continue;

									#pragma omp critical(rDs_mul)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a2b2.insert(Aa2.first, {Ab2.first, (Ab2.second - Aa2.second) % period}, D_a2b2);
										rDs_mul.insert_1st(Aa01, Ab2, {D_a2b2.shape[1], D_a.shape[0], D_a.shape[1]});
										dim_0.input(D_a2b2.shape[1], {D_a.shape[0], D_a.shape[1]}, D_a2b2.shape[0]);
									}
									rDs_a.insert_data(D_a);
									rDs_a2b2.insert_data(D_a2b2);
								} // end for Ab2
							} // end for Aa2
						} // end for Aa01
					} // end omp parallel

					rDs_a.upload(queue);
					rDs_a2b2.upload(queue);
					rDs_mul.upload_1st(queue);
					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a2b2.d_array, rDs_a.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_1,
						rDs_mul.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul * D_a1b0 * D_b
					GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_a1b0;

					#pragma omp parallel
					{
						for (std::size_t ia01 = 0; ia01 < list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01)
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const Tensor<Tdata> &D_a1b0 = tools.get_Ds_ab(Label::ab::a1b0, Aa01, Ab01);
								if(D_a1b0.empty())	continue;
								for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2)
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if (this->filter_atom->filter_for32(label, Aa01, Ab01, Ab2))	continue;
									const Tensor<Tdata> &D_b = Global_Func::find(Ds_b_transpose, Ab01.first, TAC{Ab2.first, (Ab2.second-Ab01.second)%this->period});
									if(D_b.empty())	continue;
									const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab2);
									if (!D_mul.exist)	continue;

									#pragma omp critical(rDs_output)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a1b0.insert(Aa01, Ab01, D_a1b0);
										rDs_mul.insert_2nd(D_mul);
										const GPU_Data::Pack &D_tmp = rDs_tmp.insert({D_a1b0.shape[1], D_mul.shape[0], D_mul.shape[1]});
										rDs_output.insert(Aa01, Ab01, {D_tmp.shape[2], D_b.shape[0]});
										dim_1.input(D_a1b0.shape[1], {D_mul.shape[0], D_mul.shape[1]}, D_a1b0.shape[0]);
										dim_2.input(D_tmp.shape[2], D_b.shape[0], {D_b.shape[1], D_b.shape[2]});
									}
									rDs_b.insert_data(D_b);
									rDs_a1b0.insert_data(D_a1b0);
								} // end for Aa01
							} // end for Ab2
						} // end for Ab01
					} // end omp parallel

					rDs_b.upload(queue);
					rDs_a1b0.upload(queue);
					rDs_mul.upload_2nd(queue);
					const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
					rDs_output.upload(queue);

					dim_1.upload(queue);
					dim_2.upload(queue);

					constexpr bool C0_left = true;
					GPU_Backend::gemmVbatched2s(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a1b0.d_array, rDs_mul.d_array_2,
						GPU_Backend::Type_to_GPU<Tdata>{0}, rDs_tmp.d_array,
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_b.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						C0_left,
						rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a1b0_a2b2

				case Label::ab_ab::a1b1_a2b2:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b1).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b2).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a1b1).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a2b2).index_Ds_ab[0]);

					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_a2b2;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
					GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// b2a0a1 = a2b2 * a0a1a2
					// D_mul = D_a * D_a2b2
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for1(label,Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ia2 = 0; ia2 < list_Aa2.size(); ++ia2)
							{
								const TAC &Aa2 = list_Aa2[ia2];
								if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if(this->filter_atom->filter_for31(label,Aa01,Aa2,Ab2))	continue;
									const Tensor<Tdata> &D_a2b2 = tools.get_Ds_ab(Label::ab::a2b2, Aa2, Ab2);
									if(D_a2b2.empty())	continue;

									#pragma omp critical(rDs_mul)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a2b2.insert(Aa2.first, {Ab2.first, (Ab2.second - Aa2.second) % period}, D_a2b2);
										rDs_mul.insert_1st(Aa01, Ab2, {D_a2b2.shape[1], D_a.shape[0], D_a.shape[1]});
										dim_0.input(D_a2b2.shape[1], {D_a.shape[0], D_a.shape[1]}, D_a2b2.shape[0]);
									}
									rDs_a.insert_data(D_a);
									rDs_a2b2.insert_data(D_a2b2);
								} // end for Ab2
							} // end for Aa2
						} // end for Aa01
					} // end omp parallel

					rDs_a.upload(queue);
					rDs_a2b2.upload(queue);
					rDs_mul.upload_1st(queue);
					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a2b2.d_array, rDs_a.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul.d_array_1,
						rDs_mul.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul * D_a1b1 * D_b
					GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_a1b1;

					#pragma omp parallel
					{
						for (std::size_t ia01 = 0; ia01 < list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01)
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const Tensor<Tdata> &D_a1b1 = tools.get_Ds_ab(Label::ab::a1b1, Aa01, Ab01);
								if(D_a1b1.empty())	continue;
								for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2)
								{
									const TAC &Ab2 = list_Ab2[ib2];
									if (this->filter_atom->filter_for32(label, Aa01, Ab01, Ab2))	continue;
									const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
									if(D_b.empty())	continue;
									const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab2);
									if (!D_mul.exist)	continue;

									#pragma omp critical(rDs_output)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a1b1.insert(Aa01, Ab01, D_a1b1);
										rDs_mul.insert_2nd(D_mul);
										const GPU_Data::Pack &D_tmp = rDs_tmp.insert({D_a1b1.shape[1], D_mul.shape[0], D_mul.shape[1]});
										rDs_output.insert(Aa01, Ab01, {D_tmp.shape[2], D_b.shape[0]});
										dim_1.input(D_a1b1.shape[1], {D_mul.shape[0], D_mul.shape[1]}, D_a1b1.shape[0]);
										dim_2.input(D_tmp.shape[2], D_b.shape[0], {D_b.shape[1], D_b.shape[2]});
									}
									rDs_b.insert_data(D_b);
									rDs_a1b1.insert_data(D_a1b1);
								} // end for Aa01
							} // end for Ab2
						} // end for Ab01
					} // end omp parallel

					rDs_b.upload(queue);
					rDs_a1b1.upload(queue);
					rDs_mul.upload_2nd(queue);
					const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
					rDs_output.upload(queue);

					dim_1.upload(queue);
					dim_2.upload(queue);

					constexpr bool C0_left = true;
					GPU_Backend::gemmVbatched2s(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a1b1.d_array, rDs_mul.d_array_2,
						GPU_Backend::Type_to_GPU<Tdata>{0}, rDs_tmp.d_array,
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_b.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						C0_left,
						rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a1b1_a2b2

			  // Aab_Aab::a01b2_a2b01

				case Label::ab_ab::a1b2_a2b1:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b1).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a2b1).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a1b2).index_Ds_ab[0]);

					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_b, rDs_a1b2, rDs_a2b1;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul_1, rDs_mul_2;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// D_mul1 = D_b * D_a1b2
					// b0b1a1 = b0b1b2 * a1b2
					#pragma omp parallel
					{
						for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for1(label,Ab01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
							{
								const TAC &Ab2 = list_Ab2[ib2];
								if(this->filter_atom->filter_for2(label,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
								{
									const TA &Aa01 = list_Aa01[ia01];
									if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
									const Tensor<Tdata> &D_a1b2 = tools.get_Ds_ab(Label::ab::a1b2, Aa01, Ab2);
									if(D_a1b2.empty())	continue;

									#pragma omp critical(rDs_mul_1)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a1b2.insert(Aa01, Ab2, D_a1b2);
										rDs_mul_1.insert_1st(Aa01, Ab01, {D_b.shape[0], D_b.shape[1], D_a1b2.shape[0]});
										dim_0.input({D_b.shape[0], D_b.shape[1]}, D_a1b2.shape[0], D_b.shape[2]);
									}
									rDs_b.insert_data(D_b);
									rDs_a1b2.insert_data(D_a1b2);
								} // end for Ab2
							} // end for Ab01
						} // end for Aa01
					} // end omp parallel

					rDs_b.upload(queue);
					rDs_a1b2.upload(queue);
					rDs_mul_1.upload_1st(queue);
					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::NoTrans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_b.d_array, rDs_a1b2.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul_1.d_array_1,
						rDs_mul_1.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_mul2 = D_a2b1 * D_a
					// b1a1a0 = a2b1 * a1a0a2
					#pragma omp parallel
					{
						for (std::size_t ia01 = 0; ia01 < list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for1(label,Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ia2=0; ia2<list_Aa2.size(); ++ia2)
							{
								const TAC &Aa2 = list_Aa2[ia2];
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
								for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
								{
									const TAC &Ab01 = list_Ab01[ib01];
									if(this->filter_atom->filter_for32(label, Aa01, Ab01, Aa2))	continue;
									const Tensor<Tdata> &D_a2b1 = tools.get_Ds_ab(Label::ab::a2b1, Aa2, Ab01);
									if(D_a2b1.empty())	continue;

									#pragma omp critical(rDs_mul_2)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a2b1.insert(Aa2.first, {Ab01.first, (Ab01.second - Aa2.second) % period}, D_a2b1);
										rDs_mul_2.insert_1st(Aa01, Ab01, {D_a2b1.shape[1], D_a.shape[0], D_a.shape[1]});
										dim_1.input(D_a2b1.shape[1], {D_a.shape[0], D_a.shape[1]}, D_a2b1.shape[0]);
									}
									rDs_a.insert_data(D_a);
									rDs_a2b1.insert_data(D_a2b1);
								} // end for Aa2
							} // end for Aa01
						} // end for Ab01
					} // end omp parallel

					rDs_a.upload(queue);
					rDs_a2b1.upload(queue);
					rDs_mul_2.upload_1st(queue);
					dim_1.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a2b1.d_array, rDs_a.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul_2.d_array_1,
						rDs_mul_2.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul2 * D_mul1
					// a0b0 = b1a1a0 * b0b1a1
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const GPU_Data::Pack &D_mul_1 = rDs_mul_1.find_2nd(Aa01, Ab01);
								if (!D_mul_1.exist)	continue;
								const GPU_Data::Pack &D_mul_2 = rDs_mul_2.find_2nd(Aa01, Ab01);
								if (!D_mul_2.exist)	continue;

								#pragma omp critical(rDs_output)
								{
									rDs_mul_1.insert_2nd(D_mul_1);
									rDs_mul_2.insert_2nd(D_mul_2);
									rDs_output.insert(Aa01, Ab01, {D_mul_2.shape[2], D_mul_1.shape[0]});
									dim_2.input(D_mul_2.shape[2], D_mul_1.shape[0], {D_mul_1.shape[1], D_mul_1.shape[2]});
								}
							} // end for Ab01
						} // end for Aa01
					} // end omp parallel

					rDs_mul_1.upload_2nd(queue);
					rDs_mul_2.upload_2nd(queue);
					rDs_output.upload(queue);
					dim_2.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul_2.d_array_2, rDs_mul_1.d_array_2,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						rDs_output.h_array.size(), queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a1b2_a2b1

				case Label::ab_ab::a0b2_a2b0:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b0).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a2b0).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a0b2).index_Ds_ab[0]);

					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_b, rDs_a0b2, rDs_a2b0;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul_1, rDs_mul_2;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// D_mul1 = D_b * D_a0b2
					// a0b0b1 = a0b2 * b0b1b2
					#pragma omp parallel
					{
						for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for1(label,Ab01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
							{
								const TAC &Ab2 = list_Ab2[ib2];
								if(this->filter_atom->filter_for2(label,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
								{
									const TA &Aa01 = list_Aa01[ia01];
									if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
									const Tensor<Tdata> &D_a0b2 = tools.get_Ds_ab(Label::ab::a0b2, Aa01, Ab2);
									if(D_a0b2.empty())	continue;

									#pragma omp critical(rDs_mul_1)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a0b2.insert(Aa01, Ab2, D_a0b2);
										rDs_mul_1.insert_1st(Aa01, Ab01, {D_a0b2.shape[0], D_b.shape[0], D_b.shape[1]});
										dim_0.input(D_a0b2.shape[0], {D_b.shape[0], D_b.shape[1]}, D_a0b2.shape[1]);
									}
									rDs_b.insert_data(D_b);
									rDs_a0b2.insert_data(D_a0b2);
								} // end for Ab2
							} // end for Ab01
						} // end for Aa01
					} // end omp parallel

					rDs_b.upload(queue);
					rDs_a0b2.upload(queue);
					rDs_mul_1.upload_1st(queue);
					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::NoTrans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a0b2.d_array, rDs_b.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul_1.d_array_1,
						rDs_mul_1.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_mul2 = D_a2b0 * D_a
					// a1a0b0 = a1a0a2 * a2b0
					#pragma omp parallel
					{
						for (std::size_t ia01 = 0; ia01 < list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for1(label,Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ia2=0; ia2<list_Aa2.size(); ++ia2)
							{
								const TAC &Aa2 = list_Aa2[ia2];
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
								for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
								{
									const TAC &Ab01 = list_Ab01[ib01];
									if(this->filter_atom->filter_for32(label, Aa01, Ab01, Aa2))	continue;
									const Tensor<Tdata> &D_a2b0 = tools.get_Ds_ab(Label::ab::a2b0, Aa2, Ab01);
									if(D_a2b0.empty())	continue;

									#pragma omp critical(rDs_mul_2)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a2b0.insert(Aa2.first, {Ab01.first, (Ab01.second - Aa2.second) % period}, D_a2b0);
										rDs_mul_2.insert_1st(Aa01, Ab01, {D_a.shape[0], D_a.shape[1], D_a2b0.shape[1]});
										dim_1.input({D_a.shape[0], D_a.shape[1]}, D_a2b0.shape[1], D_a.shape[2]);
									}
									rDs_a.insert_data(D_a);
									rDs_a2b0.insert_data(D_a2b0);
								} // end for Aa2
							} // end for Aa01
						} // end for Ab01
					} // end omp parallel

					rDs_a.upload(queue);
					rDs_a2b0.upload(queue);
					rDs_mul_2.upload_1st(queue);
					dim_1.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::NoTrans, GPU_Backend::NoTrans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a.d_array, rDs_a2b0.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul_2.d_array_1,
						rDs_mul_2.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul2 * D_mul1
					// a1b1 = a1a0b0 * a0b0b1
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const GPU_Data::Pack &D_mul_1 = rDs_mul_1.find_2nd(Aa01, Ab01);
								if (!D_mul_1.exist)	continue;
								const GPU_Data::Pack &D_mul_2 = rDs_mul_2.find_2nd(Aa01, Ab01);
								if (!D_mul_2.exist)	continue;

								#pragma omp critical(rDs_output)
								{
									rDs_mul_1.insert_2nd(D_mul_1);
									rDs_mul_2.insert_2nd(D_mul_2);
									rDs_output.insert(Aa01, Ab01, {D_mul_2.shape[0], D_mul_1.shape[2]});
									dim_2.input(D_mul_2.shape[0], D_mul_1.shape[2], {D_mul_1.shape[0], D_mul_1.shape[1]});
								}
							} // end for Ab01
						} // end for Aa01
					} // end omp parallel

					rDs_mul_1.upload_2nd(queue);
					rDs_mul_2.upload_2nd(queue);
					rDs_output.upload(queue);
					dim_2.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::NoTrans, GPU_Backend::NoTrans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul_2.d_array_2, rDs_mul_1.d_array_2,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						rDs_output.h_array.size(), queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a0b2_a2b0

				case Label::ab_ab::a0b2_a2b1:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b1).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a2b1).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a0b2).index_Ds_ab[0]);

					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_b, rDs_a0b2, rDs_a2b1;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul_1, rDs_mul_2;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// D_mul1 = D_b * D_a0b2
					// b0b1a0 = b0b1b2 * a0b2
					#pragma omp parallel
					{
						for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for1(label,Ab01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
							{
								const TAC &Ab2 = list_Ab2[ib2];
								if(this->filter_atom->filter_for2(label,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
								{
									const TA &Aa01 = list_Aa01[ia01];
									if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
									const Tensor<Tdata> &D_a0b2 = tools.get_Ds_ab(Label::ab::a0b2, Aa01, Ab2);
									if(D_a0b2.empty())	continue;

									#pragma omp critical(rDs_mul_1)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a0b2.insert(Aa01, Ab2, D_a0b2);
										rDs_mul_1.insert_1st(Aa01, Ab01, {D_b.shape[0], D_b.shape[1], D_a0b2.shape[0]});
										dim_0.input({D_b.shape[0], D_b.shape[1]}, D_a0b2.shape[0], D_b.shape[2]);
									}
									rDs_b.insert_data(D_b);
									rDs_a0b2.insert_data(D_a0b2);
								} // end for Ab2
							} // end for Ab01
						} // end for Aa01
					} // end omp parallel

					rDs_b.upload(queue);
					rDs_a0b2.upload(queue);
					rDs_mul_1.upload_1st(queue);
					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::NoTrans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_b.d_array, rDs_a0b2.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul_1.d_array_1,
						rDs_mul_1.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_mul2 = D_a2b1 * D_a
					// b1a0a1 = a2b1 * a0a1a2
					#pragma omp parallel
					{
						for (std::size_t ia01 = 0; ia01 < list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for1(label,Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ia2=0; ia2<list_Aa2.size(); ++ia2)
							{
								const TAC &Aa2 = list_Aa2[ia2];
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
								for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
								{
									const TAC &Ab01 = list_Ab01[ib01];
									if(this->filter_atom->filter_for32(label, Aa01, Ab01, Aa2))	continue;
									const Tensor<Tdata> &D_a2b1 = tools.get_Ds_ab(Label::ab::a2b1, Aa2, Ab01);
									if(D_a2b1.empty())	continue;

									#pragma omp critical(rDs_mul_2)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a2b1.insert(Aa2.first, {Ab01.first, (Ab01.second - Aa2.second) % period}, D_a2b1);
										rDs_mul_2.insert_1st(Aa01, Ab01, {D_a2b1.shape[1], D_a.shape[0], D_a.shape[1]});
										dim_1.input(D_a2b1.shape[1], {D_a.shape[0], D_a.shape[1]}, D_a2b1.shape[0]);
									}
									rDs_a.insert_data(D_a);
									rDs_a2b1.insert_data(D_a2b1);
								} // end for Aa2
							} // end for Aa01
						} // end for Ab01
					} // end omp parallel

					rDs_a.upload(queue);
					rDs_a2b1.upload(queue);
					rDs_mul_2.upload_1st(queue);
					dim_1.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a2b1.d_array, rDs_a.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul_2.d_array_1,
						rDs_mul_2.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul2 * D_mul1
					// a1b0 = b1a0a1 * b0b1a0
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const GPU_Data::Pack &D_mul_1 = rDs_mul_1.find_2nd(Aa01, Ab01);
								if (!D_mul_1.exist)	continue;
								const GPU_Data::Pack &D_mul_2 = rDs_mul_2.find_2nd(Aa01, Ab01);
								if (!D_mul_2.exist)	continue;

								#pragma omp critical(rDs_output)
								{
									rDs_mul_1.insert_2nd(D_mul_1);
									rDs_mul_2.insert_2nd(D_mul_2);
									rDs_output.insert(Aa01, Ab01, {D_mul_2.shape[2], D_mul_1.shape[0]});
									dim_2.input(D_mul_2.shape[2], D_mul_1.shape[0], {D_mul_1.shape[1], D_mul_1.shape[2]});
								}
							} // end for Ab01
						} // end for Aa01
					} // end omp parallel

					rDs_mul_1.upload_2nd(queue);
					rDs_mul_2.upload_2nd(queue);
					rDs_output.upload(queue);
					dim_2.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::Trans, GPU_Backend::Trans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul_2.d_array_2, rDs_mul_1.d_array_2,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						rDs_output.h_array.size(), queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a0b2_a2b1

				case Label::ab_ab::a1b2_a2b0:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b0).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a2b0).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a1b2).index_Ds_ab[0]);

					GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_b, rDs_a1b2, rDs_a2b0;
					GPU_Data::Mul<TA, TAC, Tdata> rDs_mul_1, rDs_mul_2;
					GPU_Data::Output<TA, TAC, Tdata> rDs_output;

					Dim_mnk dim_0, dim_1, dim_2;

					// D_mul1 = D_b * D_a1b2
					// a1b0b1 = a1b2 * b0b1b2
					#pragma omp parallel
					{
						for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for1(label,Ab01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
							{
								const TAC &Ab2 = list_Ab2[ib2];
								if(this->filter_atom->filter_for2(label,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
								{
									const TA &Aa01 = list_Aa01[ia01];
									if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
									const Tensor<Tdata> &D_a1b2 = tools.get_Ds_ab(Label::ab::a1b2, Aa01, Ab2);
									if(D_a1b2.empty())	continue;

									#pragma omp critical(rDs_mul_1)
									{
										rDs_b.insert(Ab01.first, {Ab2.first, (Ab2.second - Ab01.second) % period}, D_b);
										rDs_a1b2.insert(Aa01, Ab2, D_a1b2);
										rDs_mul_1.insert_1st(Aa01, Ab01, {D_a1b2.shape[0], D_b.shape[0], D_b.shape[1]});
										dim_0.input(D_a1b2.shape[0], {D_b.shape[0], D_b.shape[1]}, D_a1b2.shape[1]);
									}
									rDs_b.insert_data(D_b);
									rDs_a1b2.insert_data(D_a1b2);
								} // end for Ab2
							} // end for Ab01
						} // end for Aa01
					} // end omp parallel

					rDs_b.upload(queue);
					rDs_a1b2.upload(queue);
					rDs_mul_1.upload_1st(queue);
					dim_0.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::NoTrans, GPU_Backend::Trans,
						dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a1b2.d_array, rDs_b.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul_1.d_array_1,
						rDs_mul_1.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_mul2 = D_a2b0 * D_a
					// a0a1b0 = a0a1a2 * a2b0
					#pragma omp parallel
					{
						for (std::size_t ia01 = 0; ia01 < list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for1(label,Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ia2=0; ia2<list_Aa2.size(); ++ia2)
							{
								const TAC &Aa2 = list_Aa2[ia2];
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
								for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
								{
									const TAC &Ab01 = list_Ab01[ib01];
									if(this->filter_atom->filter_for32(label, Aa01, Ab01, Aa2))	continue;
									const Tensor<Tdata> &D_a2b0 = tools.get_Ds_ab(Label::ab::a2b0, Aa2, Ab01);
									if(D_a2b0.empty())	continue;

									#pragma omp critical(rDs_mul_2)
									{
										rDs_a.insert(Aa01, Aa2, D_a);
										rDs_a2b0.insert(Aa2.first, {Ab01.first, (Ab01.second - Aa2.second) % period}, D_a2b0);
										rDs_mul_2.insert_1st(Aa01, Ab01, {D_a.shape[0], D_a.shape[1], D_a2b0.shape[1]});
										dim_1.input({D_a.shape[0], D_a.shape[1]}, D_a2b0.shape[1], D_a.shape[2]);
									}
									rDs_a.insert_data(D_a);
									rDs_a2b0.insert_data(D_a2b0);
								} // end for Aa2
							} // end for Aa01
						} // end for Ab01
					} // end omp parallel

					rDs_a.upload(queue);
					rDs_a2b0.upload(queue);
					rDs_mul_2.upload_1st(queue);
					dim_1.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::NoTrans, GPU_Backend::NoTrans,
						dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_a.d_array, rDs_a2b0.d_array,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul_2.d_array_1,
						rDs_mul_2.h_array_1.size(), queue);
					GPU_Backend::sync(queue);

					// D_result = D_mul2 * D_mul1
					// a0b1 = a0a1b0 * a1b0b1
					#pragma omp parallel
					{
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if (this->filter_atom->filter_for1(label, Aa01))	continue;
							#pragma omp for schedule(dynamic) nowait
							for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
								const GPU_Data::Pack &D_mul_1 = rDs_mul_1.find_2nd(Aa01, Ab01);
								if (!D_mul_1.exist)	continue;
								const GPU_Data::Pack &D_mul_2 = rDs_mul_2.find_2nd(Aa01, Ab01);
								if (!D_mul_2.exist)	continue;

								#pragma omp critical(rDs_output)
								{
									rDs_mul_1.insert_2nd(D_mul_1);
									rDs_mul_2.insert_2nd(D_mul_2);
									rDs_output.insert(Aa01, Ab01, {D_mul_2.shape[0], D_mul_1.shape[2]});
									dim_2.input(D_mul_2.shape[0], D_mul_1.shape[2], {D_mul_1.shape[0], D_mul_1.shape[1]});
								}
							} // end for Ab01
						} // end for Aa01
					} // end omp parallel

					rDs_mul_1.upload_2nd(queue);
					rDs_mul_2.upload_2nd(queue);
					rDs_output.upload(queue);
					dim_2.upload(queue);

					GPU_Backend::gemmVbatched(
						GPU_Backend::NoTrans, GPU_Backend::NoTrans,
						dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_mul_2.d_array_2, rDs_mul_1.d_array_2,
						GPU_Backend::Type_to_GPU<Tdata>{1}, rDs_output.d_array,
						rDs_output.h_array.size(), queue);
					GPU_Backend::sync(queue);

					rDs_output.download(Ds_result, queue, fac_add_Ds);
				} break; // end case a1b2_a2b0

				default:
					throw std::invalid_argument(std::string(__FILE__)+std::to_string(__LINE__));
			} // end switch(label)
		} // end for label

	}

	LRI_Cal_Aux::destroy_Ds_result(Ds_result);
	#ifdef __MKL_RI
	mkl_set_num_threads(mkl_threads);
	#endif
	//for (const auto &Ds_A : Ds_result)
	//	for (const auto &Ds_B : Ds_A.second)
	//	{
	//		std::ofstream ofs("H_" + std::to_string(Ds_A.first) + "_" + std::to_string(Ds_B.first.first) + ".txt");
	//		ofs << Ds_B.second << std::endl;
	//	}

}	// end LRI::cal_loop3()

}	// end namespace RI

#endif

