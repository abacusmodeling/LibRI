#pragma once

#include<map>
#include<unordered_map>
#include<iostream>
#include"ri/Label.h"
#include"ri/LRI.h"

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

		LRI<int,int,1,Tdata> lri;

		lri.csm.set_threshold(0);
		lri.list_Aa01 = [&]()->std::set<int>{ return {Aa01}; };
		lri.list_Aa2 = [&]()->std::set<std::pair<int,std::array<int,1>>>{ return {{Aa2,{0}}}; };
		lri.list_Ab01 = [&]()->std::set<std::pair<int,std::array<int,1>>>{ return {{Ab01,{0}}}; };
		lri.list_Ab2 = [&]()->std::set<std::pair<int,std::array<int,1>>>{ return {{Ab2,{0}}}; };

		for(const Label::ab &label : Label::array_ab)
			lri.set_tensor2(Ds_ab[label], label, 0);

		{
			lri.cal({Label::ab_ab::a0b0_a1b1});
			Tensor<Tdata> D_test({Na2,Nb2});
			FOR_ia012_ib012
				D_test(ia2,ib2) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b0][Aa01][{Ab01,{0}}](ia0,ib0)
					* Ds_ab[Label::ab::a1b1][Aa01][{Ab01,{0}}](ia1,ib1)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b0_a1b1\t"<<(lri.Ds_result[Aa2][{Ab2,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a0b1_a1b0});
			Tensor<Tdata> D_test({Na2,Nb2});
			FOR_ia012_ib012
				D_test(ia2,ib2) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b1][Aa01][{Ab01,{0}}](ia0,ib1)
					* Ds_ab[Label::ab::a1b0][Aa01][{Ab01,{0}}](ia1,ib0)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b1_a1b0\t"<<(lri.Ds_result[Aa2][{Ab2,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a0b0_a2b1});
			Tensor<Tdata> D_test({Na1,Nb2});
			FOR_ia012_ib012
				D_test(ia1,ib2) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b0][Aa01][{Ab01,{0}}](ia0,ib0)
					* Ds_ab[Label::ab::a2b1][Aa2][{Ab01,{0}}](ia2,ib1)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b0_a2b1\t"<<(lri.Ds_result[Aa01][{Ab2,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a0b1_a2b0});
			Tensor<Tdata> D_test({Na1,Nb2});
			FOR_ia012_ib012
				D_test(ia1,ib2) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b1][Aa01][{Ab01,{0}}](ia0,ib1)
					* Ds_ab[Label::ab::a2b0][Aa2][{Ab01,{0}}](ia2,ib0)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b1_a2b0\t"<<(lri.Ds_result[Aa01][{Ab2,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a1b0_a2b1});
			Tensor<Tdata> D_test({Na0,Nb2});
			FOR_ia012_ib012
				D_test(ia0,ib2) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a1b0][Aa01][{Ab01,{0}}](ia1,ib0)
					* Ds_ab[Label::ab::a2b1][Aa2][{Ab01,{0}}](ia2,ib1)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a1b0_a2b1\t"<<(lri.Ds_result[Aa01][{Ab2,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a1b1_a2b0});
			Tensor<Tdata> D_test({Na0,Nb2});
			FOR_ia012_ib012
				D_test(ia0,ib2) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a1b1][Aa01][{Ab01,{0}}](ia1,ib1)
					* Ds_ab[Label::ab::a2b0][Aa2][{Ab01,{0}}](ia2,ib0)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a1b1_a2b0\t"<<(lri.Ds_result[Aa01][{Ab2,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a0b0_a1b2});
			Tensor<Tdata> D_test({Na2,Nb1});
			FOR_ia012_ib012
				D_test(ia2,ib1) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b0][Aa01][{Ab01,{0}}](ia0,ib0)
					* Ds_ab[Label::ab::a1b2][Aa01][{Ab2,{0}}](ia1,ib2)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b0_a1b2\t"<<(lri.Ds_result[Aa2][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a0b1_a1b2});
			Tensor<Tdata> D_test({Na2,Nb0});
			FOR_ia012_ib012
				D_test(ia2,ib0) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b1][Aa01][{Ab01,{0}}](ia0,ib1)
					* Ds_ab[Label::ab::a1b2][Aa01][{Ab2,{0}}](ia1,ib2)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b1_a1b2\t"<<(lri.Ds_result[Aa2][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a0b2_a1b0});
			Tensor<Tdata> D_test({Na2,Nb1});
			FOR_ia012_ib012
				D_test(ia2,ib1) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b2][Aa01][{Ab2,{0}}](ia0,ib2)
					* Ds_ab[Label::ab::a1b0][Aa01][{Ab01,{0}}](ia1,ib0)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b2_a1b0\t"<<(lri.Ds_result[Aa2][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a0b2_a1b1});
			Tensor<Tdata> D_test({Na2,Nb0});
			FOR_ia012_ib012
				D_test(ia2,ib0) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b2][Aa01][{Ab2,{0}}](ia0,ib2)
					* Ds_ab[Label::ab::a1b1][Aa01][{Ab01,{0}}](ia1,ib1)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b2_a1b1\t"<<(lri.Ds_result[Aa2][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a0b0_a2b2});
			Tensor<Tdata> D_test({Na1,Nb1});
			FOR_ia012_ib012
				D_test(ia1,ib1) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b0][Aa01][{Ab01,{0}}](ia0,ib0)
					* Ds_ab[Label::ab::a2b2][Aa2][{Ab2,{0}}](ia2,ib2)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b0_a2b2\t"<<(lri.Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a0b1_a2b2});
			Tensor<Tdata> D_test({Na1,Nb0});
			FOR_ia012_ib012
				D_test(ia1,ib0) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b1][Aa01][{Ab01,{0}}](ia0,ib1)
					* Ds_ab[Label::ab::a2b2][Aa2][{Ab2,{0}}](ia2,ib2)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b1_a2b2\t"<<(lri.Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a0b2_a2b0});
			Tensor<Tdata> D_test({Na1,Nb1});
			FOR_ia012_ib012
				D_test(ia1,ib1) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b2][Aa01][{Ab2,{0}}](ia0,ib2)
					* Ds_ab[Label::ab::a2b0][Aa2][{Ab01,{0}}](ia2,ib0)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b2_a2b0\t"<<(lri.Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a0b2_a2b1});
			Tensor<Tdata> D_test({Na1,Nb0});
			FOR_ia012_ib012
				D_test(ia1,ib0) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a0b2][Aa01][{Ab2,{0}}](ia0,ib2)
					* Ds_ab[Label::ab::a2b1][Aa2][{Ab01,{0}}](ia2,ib1)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a0b2_a2b1\t"<<(lri.Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}		

		{
			lri.cal({Label::ab_ab::a1b0_a2b2});
			Tensor<Tdata> D_test({Na0,Nb1});
			FOR_ia012_ib012
				D_test(ia0,ib1) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a1b0][Aa01][{Ab01,{0}}](ia1,ib0)
					* Ds_ab[Label::ab::a2b2][Aa2][{Ab2,{0}}](ia2,ib2)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a1b0_a2b2\t"<<(lri.Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a1b1_a2b2});
			Tensor<Tdata> D_test({Na0,Nb0});
			FOR_ia012_ib012
				D_test(ia0,ib0) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a1b1][Aa01][{Ab01,{0}}](ia1,ib1)
					* Ds_ab[Label::ab::a2b2][Aa2][{Ab2,{0}}](ia2,ib2)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a1b1_a2b2\t"<<(lri.Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a1b2_a2b0});
			Tensor<Tdata> D_test({Na0,Nb1});
			FOR_ia012_ib012
				D_test(ia0,ib1) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a1b2][Aa01][{Ab2,{0}}](ia1,ib2)
					* Ds_ab[Label::ab::a2b0][Aa2][{Ab01,{0}}](ia2,ib0)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a1b2_a2b0\t"<<(lri.Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}

		{
			lri.cal({Label::ab_ab::a1b2_a2b1});
			Tensor<Tdata> D_test({Na0,Nb0});
			FOR_ia012_ib012
				D_test(ia0,ib0) +=
					Ds_ab[Label::ab::a][Aa01][{Aa2,{0}}](ia0,ia1,ia2)
					* Ds_ab[Label::ab::a1b2][Aa01][{Ab2,{0}}](ia1,ib2)
					* Ds_ab[Label::ab::a2b1][Aa2][{Ab01,{0}}](ia2,ib1)
					* Ds_ab[Label::ab::b][Ab01][{Ab2,{0}}](ib0,ib1,ib2);
			std::cout<<"a1b2_a2b1\t"<<(lri.Ds_result[Aa01][{Ab01,{0}}] - D_test).norm(2)<<std::endl;
			lri.Ds_result.clear();
		}
	}

}