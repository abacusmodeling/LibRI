// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include"RI/ri/Label.h"
#include"RI/ri/Label_Tools.h"
#include"RI/ri/LRI.h"
#include"RI/global/MPI_Wrapper.h"

#include"unittests/print_stl.h"
#include"unittests/global/Tensor-test.h"

#include<cmath>
#include<complex>
#include<limits>
#include<stdexcept>
#include<vector>
#include<map>
#include<unordered_map>
#include<iostream>
#include<mpi.h>

#define FOR_ia012_ib012                                		\
	for(std::size_t ia0=0; ia0<Na0; ++ia0)                     	\
		for(std::size_t ia1=0; ia1<Na1; ++ia1)                 	\
			for(std::size_t ia2=0; ia2<Na2; ++ia2)             	\
				for(std::size_t ib0=0; ib0<Nb0; ++ib0)         	\
					for(std::size_t ib1=0; ib1<Nb1; ++ib1)     	\
						for(std::size_t ib2=0; ib2<Nb2; ++ib2)


namespace LRI_Loop3_Test
{
	template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
	class Parallel_LRI_test: public RI::Parallel_LRI_Equally<TA,Tcell,Ndim,Tdata>
	{
		using TatomR = std::array<double,Ndim>;		// tmp
		void set_parallel(
			const MPI_Comm &mpi_comm_in,
			const std::map<TA,TatomR> &atomsR,
			const std::array<TatomR,Ndim> &latvec,
			const std::array<Tcell,Ndim> &period_in,
			const std::set<RI::Label::Aab_Aab> &labels) override
		{
			this->mpi_comm = mpi_comm_in;
			this->period = period_in;
			const int Aa01=1, Ab01=2, Aa2=5, Ab2=6;

			this->list_Aa01 = {Aa01};
			this->list_Aa2={{Aa2,{0}}};
			this->list_Ab01={{Ab01,{0}}};
			this->list_Ab2={{Ab2,{0}}};

			for(const RI::Label::Aab_Aab &label : labels)
			{
				this->list_A[label].a01 = {Aa01};
				this->list_A[label].a2  = {{Aa2,{0}}};
				this->list_A[label].b01 = {{Ab01,{0}}};
				this->list_A[label].b2  = {{Ab2,{0}}};
			}
		}
	};

	template<typename Tdata>
	RI::Tensor<Tdata> init_tensor(const RI::Shape_Vector &shape)
	{
		RI::Tensor<Tdata> D(shape);
		for(std::size_t i=0; i<D.data->size(); ++i)
			(*D.data)[i] = i;
		return D;
	}

	template<typename Tdata>
	void test_loop3(const RI::LRI_Cal_Mode cal_mode)
	{

		auto print_result = [](const std::string &label, const RI::Tensor<Tdata> &D1, const RI::Tensor<Tdata> &D2)
		{
			std::cout<<label<<"\t"<<(D1 - D2).norm(2)<<std::endl;
			//std::cout<<"@ "<<label<<std::endl;
			//std::cout<<D1<<std::endl;
			//std::cout<<D2<<std::endl<<std::endl;
		};

		constexpr std::size_t Ndim = 1;
		const std::size_t Na0=2, Nb0=3, Na1=4, Nb1=5, Na2=6, Nb2=7;
		const int Aa01=1, Ab01=2, Aa2=5, Ab2=6;
		using T_Ds = std::map<int, std::map<std::pair<int,std::array<int,Ndim>>, RI::Tensor<Tdata>>>;
		std::unordered_map<RI::Label::ab, T_Ds> Ds_ab;
		Ds_ab.reserve(11);
		Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}] = init_tensor<Tdata>({Na0,Na1,Na2});
		Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}] = init_tensor<Tdata>({Nb0,Nb1,Nb2});
		Ds_ab[RI::Label::ab::a0b0][Aa01][{Ab01,{0}}] = init_tensor<Tdata>({Na0,Nb0});
		Ds_ab[RI::Label::ab::a0b1][Aa01][{Ab01,{0}}] = init_tensor<Tdata>({Na0,Nb1});
		Ds_ab[RI::Label::ab::a0b2][Aa01][{Ab2,{0}}] = init_tensor<Tdata>({Na0,Nb2});
		Ds_ab[RI::Label::ab::a1b0][Aa01][{Ab01,{0}}] = init_tensor<Tdata>({Na1,Nb0});
		Ds_ab[RI::Label::ab::a1b1][Aa01][{Ab01,{0}}] = init_tensor<Tdata>({Na1,Nb1});
		Ds_ab[RI::Label::ab::a1b2][Aa01][{Ab2,{0}}] = init_tensor<Tdata>({Na1,Nb2});
		Ds_ab[RI::Label::ab::a2b0][Aa2][{Ab01,{0}}] = init_tensor<Tdata>({Na2,Nb0});
		Ds_ab[RI::Label::ab::a2b1][Aa2][{Ab01,{0}}] = init_tensor<Tdata>({Na2,Nb1});
		Ds_ab[RI::Label::ab::a2b2][Aa2][{Ab2,{0}}] = init_tensor<Tdata>({Na2,Nb2});

		RI::LRI<int,int,Ndim,Tdata> lri;
		lri.cal_mode = cal_mode;
		lri.parallel = std::make_shared<Parallel_LRI_test<int,int,Ndim,Tdata>>();
		lri.set_parallel( MPI_COMM_WORLD, {}, {}, {1}, RI::Global_Func::to_vector(RI::Label::array_ab_ab) );

		//for(const RI::Label::ab &label : RI::Label::array_ab)
		//	lri.set_tensors_map2(Ds_ab[label], {label});

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a0b0, RI::Label::ab::a1b1})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a0b0_a1b1}, Ds_result);
			RI::Tensor<Tdata> D_test({Na2,Nb2});
			FOR_ia012_ib012
				D_test(ia2,ib2) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a0b0][Aa01][{Ab01,{0}}](ia0,ib0)
					* Ds_ab[RI::Label::ab::a1b1][Aa01][{Ab01,{0}}](ia1,ib1)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a0b0_a1b1", Ds_result[Aa2][{Ab2,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a0b1, RI::Label::ab::a1b0})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a0b1_a1b0}, Ds_result);
			RI::Tensor<Tdata> D_test({Na2,Nb2});
			FOR_ia012_ib012
				D_test(ia2,ib2) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a0b1][Aa01][{Ab01,{0}}](ia0,ib1)
					* Ds_ab[RI::Label::ab::a1b0][Aa01][{Ab01,{0}}](ia1,ib0)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a0b1_a1b0", Ds_result[Aa2][{Ab2,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a0b0, RI::Label::ab::a2b1})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a0b0_a2b1}, Ds_result);
			RI::Tensor<Tdata> D_test({Na1,Nb2});
			FOR_ia012_ib012
				D_test(ia1,ib2) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a0b0][Aa01][{Ab01,{0}}](ia0,ib0)
					* Ds_ab[RI::Label::ab::a2b1][Aa2][{Ab01,{0}}](ia2,ib1)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a0b0_a2b1", Ds_result[Aa01][{Ab2,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a0b1, RI::Label::ab::a2b0})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a0b1_a2b0}, Ds_result);
			RI::Tensor<Tdata> D_test({Na1,Nb2});
			FOR_ia012_ib012
				D_test(ia1,ib2) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a0b1][Aa01][{Ab01,{0}}](ia0,ib1)
					* Ds_ab[RI::Label::ab::a2b0][Aa2][{Ab01,{0}}](ia2,ib0)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a0b1_a2b0", Ds_result[Aa01][{Ab2,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a1b0, RI::Label::ab::a2b1})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a1b0_a2b1}, Ds_result);
			RI::Tensor<Tdata> D_test({Na0,Nb2});
			FOR_ia012_ib012
				D_test(ia0,ib2) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a1b0][Aa01][{Ab01,{0}}](ia1,ib0)
					* Ds_ab[RI::Label::ab::a2b1][Aa2][{Ab01,{0}}](ia2,ib1)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a1b0_a2b1", Ds_result[Aa01][{Ab2,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a1b1, RI::Label::ab::a2b0})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a1b1_a2b0}, Ds_result);
			RI::Tensor<Tdata> D_test({Na0,Nb2});
			FOR_ia012_ib012
				D_test(ia0,ib2) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a1b1][Aa01][{Ab01,{0}}](ia1,ib1)
					* Ds_ab[RI::Label::ab::a2b0][Aa2][{Ab01,{0}}](ia2,ib0)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a1b1_a2b0", Ds_result[Aa01][{Ab2,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a0b0, RI::Label::ab::a1b2})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a0b0_a1b2}, Ds_result);
			RI::Tensor<Tdata> D_test({Na2,Nb1});
			FOR_ia012_ib012
				D_test(ia2,ib1) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a0b0][Aa01][{Ab01,{0}}](ia0,ib0)
					* Ds_ab[RI::Label::ab::a1b2][Aa01][{Ab2,{0}}](ia1,ib2)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a0b0_a1b2", Ds_result[Aa2][{Ab01,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a0b1, RI::Label::ab::a1b2})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a0b1_a1b2}, Ds_result);
			RI::Tensor<Tdata> D_test({Na2,Nb0});
			FOR_ia012_ib012
				D_test(ia2,ib0) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a0b1][Aa01][{Ab01,{0}}](ia0,ib1)
					* Ds_ab[RI::Label::ab::a1b2][Aa01][{Ab2,{0}}](ia1,ib2)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a0b1_a1b2", Ds_result[Aa2][{Ab01,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a0b2, RI::Label::ab::a1b0})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a0b2_a1b0}, Ds_result);
			RI::Tensor<Tdata> D_test({Na2,Nb1});
			FOR_ia012_ib012
				D_test(ia2,ib1) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a0b2][Aa01][{Ab2,{0}}](ia0,ib2)
					* Ds_ab[RI::Label::ab::a1b0][Aa01][{Ab01,{0}}](ia1,ib0)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a0b2_a1b0", Ds_result[Aa2][{Ab01,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a0b2, RI::Label::ab::a1b1})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a0b2_a1b1}, Ds_result);
			RI::Tensor<Tdata> D_test({Na2,Nb0});
			FOR_ia012_ib012
				D_test(ia2,ib0) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a0b2][Aa01][{Ab2,{0}}](ia0,ib2)
					* Ds_ab[RI::Label::ab::a1b1][Aa01][{Ab01,{0}}](ia1,ib1)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a0b2_a1b1", Ds_result[Aa2][{Ab01,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a0b0, RI::Label::ab::a2b2})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a0b0_a2b2}, Ds_result);
			RI::Tensor<Tdata> D_test({Na1,Nb1});
			FOR_ia012_ib012
				D_test(ia1,ib1) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a0b0][Aa01][{Ab01,{0}}](ia0,ib0)
					* Ds_ab[RI::Label::ab::a2b2][Aa2][{Ab2,{0}}](ia2,ib2)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a0b0_a2b2", Ds_result[Aa01][{Ab01,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a0b1, RI::Label::ab::a2b2})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a0b1_a2b2}, Ds_result);
			RI::Tensor<Tdata> D_test({Na1,Nb0});
			FOR_ia012_ib012
				D_test(ia1,ib0) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a0b1][Aa01][{Ab01,{0}}](ia0,ib1)
					* Ds_ab[RI::Label::ab::a2b2][Aa2][{Ab2,{0}}](ia2,ib2)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a0b1_a2b2", Ds_result[Aa01][{Ab01,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a0b2, RI::Label::ab::a2b0})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a0b2_a2b0}, Ds_result);
			RI::Tensor<Tdata> D_test({Na1,Nb1});
			FOR_ia012_ib012
				D_test(ia1,ib1) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a0b2][Aa01][{Ab2,{0}}](ia0,ib2)
					* Ds_ab[RI::Label::ab::a2b0][Aa2][{Ab01,{0}}](ia2,ib0)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a0b2_a2b0", Ds_result[Aa01][{Ab01,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a0b2, RI::Label::ab::a2b1})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a0b2_a2b1}, Ds_result);
			RI::Tensor<Tdata> D_test({Na1,Nb0});
			FOR_ia012_ib012
				D_test(ia1,ib0) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a0b2][Aa01][{Ab2,{0}}](ia0,ib2)
					* Ds_ab[RI::Label::ab::a2b1][Aa2][{Ab01,{0}}](ia2,ib1)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a0b2_a2b1", Ds_result[Aa01][{Ab01,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a1b0, RI::Label::ab::a2b2})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a1b0_a2b2}, Ds_result);
			RI::Tensor<Tdata> D_test({Na0,Nb1});
			FOR_ia012_ib012
				D_test(ia0,ib1) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a1b0][Aa01][{Ab01,{0}}](ia1,ib0)
					* Ds_ab[RI::Label::ab::a2b2][Aa2][{Ab2,{0}}](ia2,ib2)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a1b0_a2b2", Ds_result[Aa01][{Ab01,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a1b1, RI::Label::ab::a2b2})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a1b1_a2b2}, Ds_result);
			RI::Tensor<Tdata> D_test({Na0,Nb0});
			FOR_ia012_ib012
				D_test(ia0,ib0) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a1b1][Aa01][{Ab01,{0}}](ia1,ib1)
					* Ds_ab[RI::Label::ab::a2b2][Aa2][{Ab2,{0}}](ia2,ib2)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a1b1_a2b2", Ds_result[Aa01][{Ab01,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a1b2, RI::Label::ab::a2b0})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a1b2_a2b0}, Ds_result);
			RI::Tensor<Tdata> D_test({Na0,Nb1});
			FOR_ia012_ib012
				D_test(ia0,ib1) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a1b2][Aa01][{Ab2,{0}}](ia1,ib2)
					* Ds_ab[RI::Label::ab::a2b0][Aa2][{Ab01,{0}}](ia2,ib0)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a1b2_a2b0", Ds_result[Aa01][{Ab01,{0}}], D_test);
		}

		{
			lri.data_ab_name.clear();	lri.data_pool.clear();
			for(const RI::Label::ab &label : {RI::Label::ab::a, RI::Label::ab::b, RI::Label::ab::a1b2, RI::Label::ab::a2b1})
				lri.set_tensors_map2(Ds_ab[label], {label});
			T_Ds Ds_result;
			lri.cal_loop3({RI::Label::ab_ab::a1b2_a2b1}, Ds_result);
			RI::Tensor<Tdata> D_test({Na0,Nb0});
			FOR_ia012_ib012
				D_test(ia0,ib0) +=
					Ds_ab[RI::Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[RI::Label::ab::a1b2][Aa01][{Ab2,{0}}](ia1,ib2)
					* Ds_ab[RI::Label::ab::a2b1][Aa2][{Ab01,{0}}](ia2,ib1)
					* Ds_ab[RI::Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			print_result("a1b2_a2b1", Ds_result[Aa01][{Ab01,{0}}], D_test);
		}
	}
	

	template<typename Tdata>
	void test_loop3_accumulation(const RI::LRI_Cal_Mode cal_mode)
	{
		using LRI = RI::LRI<int,int,1,Tdata>;
		using Treal = typename LRI::Tdata_real;
		using T_Ds = std::map<int, std::map<typename LRI::TAC, RI::Tensor<Tdata>>>;
		const typename LRI::TAC key = {0,{0}};
		const auto labels = RI::Global_Func::to_vector(RI::Label::array_ab_ab);
		const auto make_tensor = [](const RI::Shape_Vector &shape)
		{
			RI::Tensor<Tdata> D(shape);
			for(std::size_t i=0; i<D.data->size(); ++i)
				(*D.data)[i] = RI::Global_Func::convert<Tdata>(
					std::complex<Treal>(0.5+0.07*i, 0.1+0.03*i));
			return D;
		};

		LRI lri;
		lri.set_parallel(MPI_COMM_SELF, {{0,{0.0}}}, {{{1.0}}}, {1}, labels);
		for(const RI::Label::ab label : RI::Label::array_ab)
		{
			const bool coefficient = label==RI::Label::ab::a || label==RI::Label::ab::b;
			T_Ds Ds;
			Ds[0][key] = make_tensor(coefficient ? RI::Shape_Vector{2,2,2} : RI::Shape_Vector{2,2});
			lri.set_tensors_map2(Ds, {label});
		}

		const RI::Tensor<Tdata> seed = make_tensor({2,2});
		const Treal tolerance = 128*std::numeric_limits<Treal>::epsilon();
		std::vector<std::vector<RI::Label::ab_ab>> label_groups;
		for(const RI::Label::ab_ab label : labels)
			label_groups.push_back({label});
		label_groups.push_back(labels);

		for(const auto &cal_labels : label_groups)
		{
			T_Ds reference;
			lri.cal_mode = RI::LRI_Cal_Mode::CPU;
			lri.cal_loop3(cal_labels, reference);
			const RI::Tensor<Tdata> contribution = reference.at(0).at(key).copy();
			const Treal norm = contribution.norm(2);
			if(!std::isfinite(norm) || norm==0)
				throw std::runtime_error("Invalid CPU reference contraction");

			lri.cal_mode = cal_mode;
			for(const bool seeded : {false, true})
			{
				T_Ds result;
				// Tensor copies share storage unless copy() is used explicitly.
				if(seeded)
					result[0][key] = seed.copy();
				Treal accumulated_factor = 0, accumulated_magnitude = 0;
				for(const double factor : {0.0, -1.0, 1.0, 1.0, -1.0, 0.5, 0.0, -0.5, 0.3})
				{
					lri.cal_loop3(cal_labels, result, factor);
					accumulated_factor += Treal(factor);
					accumulated_magnitude += std::abs(Treal(factor));
					const RI::Tensor<Tdata> &actual = result.at(0).at(key);
					if(result.size()!=1 || result.at(0).size()!=1 || actual.shape.size()!=2
						|| actual.shape[0]!=2 || actual.shape[1]!=2)
						throw std::runtime_error("Unexpected contraction output shape");
					for(std::size_t i=0; i<seed.data->size(); ++i)
					{
						const Tdata initial = seeded ? (*seed.data)[i] : Tdata(0);
						const Tdata expected = initial + accumulated_factor*(*contribution.data)[i];
						const Treal error = std::abs((*actual.data)[i] - expected);
						const Treal scale = 1 + std::abs(initial)
							+ accumulated_magnitude*std::abs((*contribution.data)[i]);
						if(!std::isfinite(error) || error>tolerance*scale)
						{
							std::cerr<<RI::Label_Tools::get_name(cal_labels)
								<<" mode="<<static_cast<int>(cal_mode)<<" seeded="<<seeded
								<<" factor="<<factor<<" element="<<i<<" error="<<error<<std::endl;
							throw std::runtime_error("Accumulation differs from the CPU reference");
						}
					}
				}
			}
		}
		std::cout<<"Accumulation: all 18 contractions and their sum passed, mode="<<static_cast<int>(cal_mode)<<std::endl;
	}


	template<typename Tdata>
	void main(int argc, char *argv[])
	{
		int mpi_init_provide;
		MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_init_provide);

		const std::vector<RI::LRI_Cal_Mode> cal_modes = {
			RI::LRI_Cal_Mode::CPU,
			RI::LRI_Cal_Mode::CPU_fine_grained_lock,
			#ifdef __GPU_RI
			RI::LRI_Cal_Mode::GPU,
			#endif
		};
		for(const RI::LRI_Cal_Mode cal_mode: cal_modes)
		{
			test_loop3<Tdata>(cal_mode);
			test_loop3_accumulation<Tdata>(cal_mode);
		}

		MPI_Finalize();
	}

}

#undef FOR_ia012_ib012
