#pragma once

#include<map>
#include<unordered_map>
#include<iostream>
#include<mpi.h>
#include"ri/Label.h"
#include"ri/LRI.h"
#include"global/MPI_Wrapper.h"

#include"unittests/print_stl.h"
#include"unittests/global/Tensor-test.h"

#define FOR_ia012_ib012                                		\
	for(size_t ia0=0; ia0<Na0; ++ia0)                     	\
		for(size_t ia1=0; ia1<Na1; ++ia1)                 	\
			for(size_t ia2=0; ia2<Na2; ++ia2)             	\
				for(size_t ib0=0; ib0<Nb0; ++ib0)         	\
					for(size_t ib1=0; ib1<Nb1; ++ib1)     	\
						for(size_t ib2=0; ib2<Nb2; ++ib2)


namespace LRI_Test
{
	template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
	class Parallel_LRI_test: public Parallel_LRI_Equally<TA,Tcell,Ndim,Tdata>
	{
		using TatomR = std::array<double,Ndim>;		// tmp
		void set_parallel(
			const MPI_Comm &mpi_comm_in,
			const std::map<TA,TatomR> &atomsR,
			const std::array<TatomR,Ndim> &latvec,
			const std::array<Tcell,Ndim> &period_in) override
		{
			this->mpi_comm = mpi_comm_in;
			this->period = period_in;
			const int Aa01=1, Ab01=2, Aa2=5, Ab2=6;
			this->list_Aa01 = {Aa01};
			this->list_Aa2={{Aa2,{0}}};
			this->list_Ab01={{Ab01,{0}}};
			this->list_Ab2={{Ab2,{0}}};
		}
	};

	template<typename Tdata>
	static Tensor<Tdata> init_tensor(const std::vector<size_t> &shape)
	{
		Tensor<Tdata> D(shape);
		for(size_t i=0; i<D.data->size(); ++i)
			(*D.data)[i] = i;
		return D;
	}

	template<typename Tdata>
	void main()
	{
		int mpi_init_provide;
		MPI_Init_thread(NULL,NULL, MPI_THREAD_MULTIPLE, &mpi_init_provide);

		if(MPI_Wrapper::mpi_get_rank(MPI_COMM_WORLD)!=0)
		{
			MPI_Finalize();
			return;
		}

		const size_t Na0=2, Nb0=3, Na1=4, Nb1=5, Na2=6, Nb2=7;
		const int Aa01=1, Ab01=2, Aa2=5, Ab2=6;
		std::unordered_map<Label::ab, std::map<int, std::map<std::pair<int,std::array<int,1>>, Tensor<Tdata>>>> Ds_ab;
		Ds_ab.reserve(11);
		Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}] = init_tensor<Tdata>({Na0,Na1,Na2});
		Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}] = init_tensor<Tdata>({Nb0,Nb1,Nb2});
		Ds_ab[Label::ab::a0b0][Aa01][{Ab01,{0}}] = init_tensor<Tdata>({Na0,Nb0});
		Ds_ab[Label::ab::a0b1][Aa01][{Ab01,{0}}] = init_tensor<Tdata>({Na0,Nb1});
		Ds_ab[Label::ab::a0b2][Aa01][{Ab2,{0}}] = init_tensor<Tdata>({Na0,Nb2});
		Ds_ab[Label::ab::a1b0][Aa01][{Ab01,{0}}] = init_tensor<Tdata>({Na1,Nb0});
		Ds_ab[Label::ab::a1b1][Aa01][{Ab01,{0}}] = init_tensor<Tdata>({Na1,Nb1});
		Ds_ab[Label::ab::a1b2][Aa01][{Ab2,{0}}] = init_tensor<Tdata>({Na1,Nb2});
		Ds_ab[Label::ab::a2b0][Aa2][{Ab01,{0}}] = init_tensor<Tdata>({Na2,Nb0});
		Ds_ab[Label::ab::a2b1][Aa2][{Ab01,{0}}] = init_tensor<Tdata>({Na2,Nb1});
		Ds_ab[Label::ab::a2b2][Aa2][{Ab2,{0}}] = init_tensor<Tdata>({Na2,Nb2});

		constexpr size_t Ndim = 1;
		LRI<int,int,Ndim,Tdata> lri(MPI_COMM_WORLD);
		lri.parallel = std::make_shared<Parallel_LRI_test<int,int,Ndim,Tdata>>();
		lri.set_parallel({}, {}, {1});

		lri.csm.set_threshold(0);

		for(const Label::ab &label : Label::array_ab)
			lri.set_tensors_map2(Ds_ab[label], label, 0);
		{
			auto Ds_result = lri.cal({Label::ab_ab::a0b0_a1b1});
			Tensor<Tdata> D_test({Na2,Nb2});
			FOR_ia012_ib012
				D_test(ia2,ib2) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b0][Aa01][{Ab01,{0}}](ia0,ib0)
					* Ds_ab[Label::ab::a1b1][Aa01][{Ab01,{0}}](ia1,ib1)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b0_a1b1\t"<<(Ds_result[Aa2][{Ab2,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a0b1_a1b0});
			Tensor<Tdata> D_test({Na2,Nb2});
			FOR_ia012_ib012
				D_test(ia2,ib2) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b1][Aa01][{Ab01,{0}}](ia0,ib1)
					* Ds_ab[Label::ab::a1b0][Aa01][{Ab01,{0}}](ia1,ib0)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b1_a1b0\t"<<(Ds_result[Aa2][{Ab2,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a0b0_a2b1});
			Tensor<Tdata> D_test({Na1,Nb2});
			FOR_ia012_ib012
				D_test(ia1,ib2) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b0][Aa01][{Ab01,{0}}](ia0,ib0)
					* Ds_ab[Label::ab::a2b1][Aa2][{Ab01,{0}}](ia2,ib1)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b0_a2b1\t"<<(Ds_result[Aa01][{Ab2,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a0b1_a2b0});
			Tensor<Tdata> D_test({Na1,Nb2});
			FOR_ia012_ib012
				D_test(ia1,ib2) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b1][Aa01][{Ab01,{0}}](ia0,ib1)
					* Ds_ab[Label::ab::a2b0][Aa2][{Ab01,{0}}](ia2,ib0)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b1_a2b0\t"<<(Ds_result[Aa01][{Ab2,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a1b0_a2b1});
			Tensor<Tdata> D_test({Na0,Nb2});
			FOR_ia012_ib012
				D_test(ia0,ib2) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a1b0][Aa01][{Ab01,{0}}](ia1,ib0)
					* Ds_ab[Label::ab::a2b1][Aa2][{Ab01,{0}}](ia2,ib1)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a1b0_a2b1\t"<<(Ds_result[Aa01][{Ab2,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a1b1_a2b0});
			Tensor<Tdata> D_test({Na0,Nb2});
			FOR_ia012_ib012
				D_test(ia0,ib2) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a1b1][Aa01][{Ab01,{0}}](ia1,ib1)
					* Ds_ab[Label::ab::a2b0][Aa2][{Ab01,{0}}](ia2,ib0)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a1b1_a2b0\t"<<(Ds_result[Aa01][{Ab2,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a0b0_a1b2});
			Tensor<Tdata> D_test({Na2,Nb1});
			FOR_ia012_ib012
				D_test(ia2,ib1) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b0][Aa01][{Ab01,{0}}](ia0,ib0)
					* Ds_ab[Label::ab::a1b2][Aa01][{Ab2,{0}}](ia1,ib2)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b0_a1b2\t"<<(Ds_result[Aa2][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a0b1_a1b2});
			Tensor<Tdata> D_test({Na2,Nb0});
			FOR_ia012_ib012
				D_test(ia2,ib0) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b1][Aa01][{Ab01,{0}}](ia0,ib1)
					* Ds_ab[Label::ab::a1b2][Aa01][{Ab2,{0}}](ia1,ib2)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b1_a1b2\t"<<(Ds_result[Aa2][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a0b2_a1b0});
			Tensor<Tdata> D_test({Na2,Nb1});
			FOR_ia012_ib012
				D_test(ia2,ib1) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b2][Aa01][{Ab2,{0}}](ia0,ib2)
					* Ds_ab[Label::ab::a1b0][Aa01][{Ab01,{0}}](ia1,ib0)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b2_a1b0\t"<<(Ds_result[Aa2][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a0b2_a1b1});
			Tensor<Tdata> D_test({Na2,Nb0});
			FOR_ia012_ib012
				D_test(ia2,ib0) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b2][Aa01][{Ab2,{0}}](ia0,ib2)
					* Ds_ab[Label::ab::a1b1][Aa01][{Ab01,{0}}](ia1,ib1)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b2_a1b1\t"<<(Ds_result[Aa2][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a0b0_a2b2});
			Tensor<Tdata> D_test({Na1,Nb1});
			FOR_ia012_ib012
				D_test(ia1,ib1) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b0][Aa01][{Ab01,{0}}](ia0,ib0)
					* Ds_ab[Label::ab::a2b2][Aa2][{Ab2,{0}}](ia2,ib2)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b0_a2b2\t"<<(Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a0b1_a2b2});
			Tensor<Tdata> D_test({Na1,Nb0});
			FOR_ia012_ib012
				D_test(ia1,ib0) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b1][Aa01][{Ab01,{0}}](ia0,ib1)
					* Ds_ab[Label::ab::a2b2][Aa2][{Ab2,{0}}](ia2,ib2)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b1_a2b2\t"<<(Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a0b2_a2b0});
			Tensor<Tdata> D_test({Na1,Nb1});
			FOR_ia012_ib012
				D_test(ia1,ib1) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b2][Aa01][{Ab2,{0}}](ia0,ib2)
					* Ds_ab[Label::ab::a2b0][Aa2][{Ab01,{0}}](ia2,ib0)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b2_a2b0\t"<<(Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a0b2_a2b1});
			Tensor<Tdata> D_test({Na1,Nb0});
			FOR_ia012_ib012
				D_test(ia1,ib0) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b2][Aa01][{Ab2,{0}}](ia0,ib2)
					* Ds_ab[Label::ab::a2b1][Aa2][{Ab01,{0}}](ia2,ib1)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b2_a2b1\t"<<(Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
		}		

		{
			auto Ds_result = lri.cal({Label::ab_ab::a1b0_a2b2});
			Tensor<Tdata> D_test({Na0,Nb1});
			FOR_ia012_ib012
				D_test(ia0,ib1) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a1b0][Aa01][{Ab01,{0}}](ia1,ib0)
					* Ds_ab[Label::ab::a2b2][Aa2][{Ab2,{0}}](ia2,ib2)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a1b0_a2b2\t"<<(Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a1b1_a2b2});
			Tensor<Tdata> D_test({Na0,Nb0});
			FOR_ia012_ib012
				D_test(ia0,ib0) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a1b1][Aa01][{Ab01,{0}}](ia1,ib1)
					* Ds_ab[Label::ab::a2b2][Aa2][{Ab2,{0}}](ia2,ib2)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a1b1_a2b2\t"<<(Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a1b2_a2b0});
			Tensor<Tdata> D_test({Na0,Nb1});
			FOR_ia012_ib012
				D_test(ia0,ib1) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a1b2][Aa01][{Ab2,{0}}](ia1,ib2)
					* Ds_ab[Label::ab::a2b0][Aa2][{Ab01,{0}}](ia2,ib0)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a1b2_a2b0\t"<<(Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
		}

		{
			auto Ds_result = lri.cal({Label::ab_ab::a1b2_a2b1});
			Tensor<Tdata> D_test({Na0,Nb0});
			FOR_ia012_ib012
				D_test(ia0,ib0) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a1b2][Aa01][{Ab2,{0}}](ia1,ib2)
					* Ds_ab[Label::ab::a2b1][Aa2][{Ab01,{0}}](ia2,ib1)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a1b2_a2b1\t"<<(Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
		}

		MPI_Finalize();
	}

}