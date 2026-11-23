# ===================
#  Author: Peize Lin
#  date: 2025.03.28
# ===================

import textwrap

get_D_a = lambda a : "tools.get_Ds_ab(Label::ab::a, Aa01, Aa2)" \
					if a=="a0a1a2" else "Global_Func::find(Ds_a_transpose, Aa01, Aa2)"
get_D_b = lambda b : "tools.get_Ds_ab(Label::ab::b, Ab01, Ab2)" \
					if b=="b0b1b2" else "Global_Func::find(Ds_b_transpose, Ab01.first, TAC{Ab2.first, (Ab2.second-Ab01.second)%this->period})"

split_tensor = lambda lst: [lst[i:i+2] for i in range(0, len(lst), 2)]

def tensor_multiply(foruma):
	prod, mulx, muly = foruma.split()[::2]
	spl = lambda lst: [lst[i:i+2] for i in range(0, len(lst), 2)]

def get_ab(formulas):
	for formula in formulas:
		prod_name, mulx_name, muly_name = formula.split()[::2]
		for D in (mulx_name, muly_name):
			if D in ["a0a1a2", "a1a0a2"]:
				a = D
			elif D in ["b0b1b2", "b1b0b2"]:
				b = D
	return a,b

def get_func(formula):
	prod_name, mulx_name, muly_name = formula.split()[::2]
	prod_split = split_tensor(prod_name)
	mulx_split = split_tensor(mulx_name)
	muly_split = split_tensor(muly_name)
	alphabet = "abcdefg"
	prod_func = [None]*len(prod_split)
	mulx_func = [None]*len(mulx_split)
	muly_func = [None]*len(muly_split)

	# 2, 1, 4 = get_equal([x0,x1,a,b,c,d,x6,...], [y0,a,b,c,d,y5,...])
	def get_equal(mulx_split, muly_split):
		for ix,ux in enumerate(mulx_split):
			for iy,uy in enumerate(muly_split):
				if ux==uy:
					for i in range(1,100):
						if ix+i==len(mulx_split) or iy+i==len(muly_split) or mulx_split[ix+i]!=muly_split[iy+i]:
							return ix,iy,i
	index_x, index_y, index_len = get_equal(mulx_split, muly_split)

	for iz1,ix in enumerate(list(range(0, index_x)) + list(range(index_x+index_len, len(mulx_split)))):
		mulx_func[ix] = prod_func[iz1] = f"x{ix}"
	for iz2,iy in enumerate(list(range(0, index_y)) + list(range(index_y+index_len, len(muly_split)))):
		muly_func[iy] = prod_func[iz1+iz2+1] = f"y{iy}"
	for i in range(0, index_len):
		mulx_func[index_x+i] = muly_func[index_y+i] = alphabet[i]
	return "_".join([
		"".join(prod_func),
		"".join(mulx_func),
		"".join(muly_func)])


def get_call(formula, other):
	prod_name, mulx_name, muly_name = formula.split()[::2]
	func = get_func(formula)
	def get_D(name):
		if len(split_tensor(name)) == 2:
			return name
		elif name in ["a0a1a2", "a1a0a2"]:
			return "a"
		elif name in ["b0b1b2", "b1b0b2"]:
			return "b"
		else:
			return other
	return f"{func}(D_{get_D(mulx_name)}, D_{get_D(muly_name)})"



def content_a01b01_a01b01(ab_ab, formulas):
	a01b01 = ab_ab.split("_")
	a, b = get_ab(formulas)
	content = textwrap.indent(textwrap.dedent(f"""\
		case Label::ab_ab::{ab_ab}:
		{{
			const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map( LRI_Cal_Aux::filter_list_map(
				list_Aa01_Da,
				data_wrapper(Label::ab::{a01b01[0]}).Ds_ab ),
				data_wrapper(Label::ab::{a01b01[1]}).Ds_ab );
			const std::vector<TAC> &list_Aa2 =
				list_Aa2_Da;
			const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set( LRI_Cal_Aux::filter_list_set(
				list_Ab01_Db,
				data_wrapper(Label::ab::{a01b01[0]}).index_Ds_ab[0]),
				data_wrapper(Label::ab::{a01b01[1]}).index_Ds_ab[0]);
			const std::vector<TAC> &list_Ab2 =
				list_Ab2_Db;

			for(const TAC &Aa2 : list_Aa2)
			{{
				if(this->filter_atom->filter_for1(label,Aa2))	continue;
				std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

				#pragma omp for schedule(dynamic) nowait
				for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
				{{
					const TAC &Ab01 = list_Ab01[ib01];
					if(this->filter_atom->filter_for2(label,Aa2,Ab01))	continue;
					// D_mul = D_a * D_{a01b01[0]} * D_{a01b01[1]}
					Tensor<Tdata> D_mul;
					for(const TA &Aa01 : list_Aa01)
					{{
						if(this->filter_atom->filter_for31(label,Aa2,Ab01,Aa01))	continue;
						const Tensor<Tdata> &D_a = {get_D_a(a)};
						if(D_a.empty())	continue;
						const Tensor<Tdata> &D_{a01b01[0]} = tools.get_Ds_ab(Label::ab::{a01b01[0]}, Aa01, Ab01);
						if(D_{a01b01[0]}.empty())	continue;
						const Tensor<Tdata> &D_{a01b01[1]} = tools.get_Ds_ab(Label::ab::{a01b01[1]}, Aa01, Ab01);
						if(D_{a01b01[1]}.empty())	continue;

						// {formulas[0]}
						const Tensor<Tdata> D_tmp1 = Tensor_Multiply::{get_call(formulas[0],None)};
						// {formulas[1]}
						Tensor<Tdata> D_tmp2 = Tensor_Multiply::{get_call(formulas[1],"tmp1")};
						LRI_Cal_Aux::add_Ds(std::move(D_tmp2), D_mul);
					}}
					if(D_mul.empty())	continue;

					// D_result = D_mul * D_b
					for(const TAC &Ab2 : list_Ab2)
					{{
						if(this->filter_atom->filter_for32(label,Aa2,Ab01,Ab2))	continue;
						const Tensor<Tdata> &D_b = {get_D_b(b)};
						if(D_b.empty())	continue;
						// {formulas[2]}
						Tensor<Tdata> D_tmp3 = Tensor_Multiply::{get_call(formulas[2],"mul")};
						LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab2]);
					}}
				}} // end for Ab01

				if(!Ds_result_fixed.empty())
					LRI_Cal_Aux::add_Ds(LRI_Cal_Aux::Ds_translate(std::move(Ds_result_fixed), Aa2.second, this->period),
										Ds_result_thread[Aa2.first]);
				LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
			}} // end for Ab2
		}} break; // end case {ab_ab}
		"""),
		"\t"*4)
	return content



def content_a01b01_a01b2(ab_ab, formulas):
	ab_ab_tmp = ab_ab.split("_")
	a01b01 = [i for i in ab_ab_tmp if not "2" in i][0]
	a01b2 = [i for i in ab_ab_tmp if "2" in i][0]
	a, b = get_ab(formulas)
	content = textwrap.indent(textwrap.dedent(f"""\
		case Label::ab_ab::{ab_ab}:
		{{
			const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map( LRI_Cal_Aux::filter_list_map(
				list_Aa01_Da,
				data_wrapper(Label::ab::{a01b01}).Ds_ab ),
				data_wrapper(Label::ab::{a01b2}).Ds_ab );
			const std::vector<TAC> &list_Aa2 =
				list_Aa2_Da;
			const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
				list_Ab01_Db,
				data_wrapper(Label::ab::{a01b01}).index_Ds_ab[0]);
			const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
				list_Ab2_Db,
				data_wrapper(Label::ab::{a01b2}).index_Ds_ab[0]);

			for(const TAC &Ab01 : list_Ab01)
			{{
				if(this->filter_atom->filter_for1(label,Ab01))	continue;
				std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

				#pragma omp for schedule(dynamic) nowait
				for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
				{{
					const TA &Aa01 = list_Aa01[ia01];
					if(this->filter_atom->filter_for2(label,Ab01,Aa01))	continue;
					const Tensor<Tdata> &D_{a01b01} = tools.get_Ds_ab(Label::ab::{a01b01}, Aa01, Ab01);
					if(D_{a01b01}.empty())	continue;
					// D_mul = D_b * D_{a01b2}
					Tensor<Tdata> D_mul;
					for(const TAC &Ab2 : list_Ab2)
					{{
						if(this->filter_atom->filter_for31(label,Ab01,Aa01,Ab2))	continue;
						const Tensor<Tdata> &D_b = {get_D_b(b)};
						if(D_b.empty())	continue;
						const Tensor<Tdata> &D_{a01b2} = tools.get_Ds_ab(Label::ab::{a01b2}, Aa01, Ab2);
						if(D_{a01b2}.empty())	continue;

						// {formulas[0]}
						Tensor<Tdata> D_tmp1 = Tensor_Multiply::{get_call(formulas[0],None)};
						LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
					}}
					if(D_mul.empty())	continue;

					// D_result = D_mul * D_a * D_{a01b01}
					for(const TAC &Aa2 : list_Aa2)
					{{
						if(this->filter_atom->filter_for32(label,Ab01,Aa01,Aa2))	continue;
						const Tensor<Tdata> &D_a = {get_D_a(a)};
						if(D_a.empty())	continue;
						// {formulas[1]}
						const Tensor<Tdata> D_tmp2 = Tensor_Multiply::{get_call(formulas[1],"mul")};
						// {formulas[2]}
						Tensor<Tdata> D_tmp3 = Tensor_Multiply::{get_call(formulas[2],"tmp2")};
						LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Aa2]);
					}}
				}} // end for Aa01

				if(!Ds_result_fixed.empty())
					LRI_Cal_Aux::add_Ds(LRI_Cal_Aux::Ds_exchange(std::move(Ds_result_fixed), Ab01, this->period),
										Ds_result_thread);
				LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
			}} // end for Ab01
		}} break; // end case {ab_ab}
		"""),
		"\t"*4)
	return content



def content_a01b01_a2b01(ab_ab, formulas):
	ab_ab_tmp = ab_ab.split("_")
	a01b01 = [i for i in ab_ab_tmp if not "2" in i][0]
	a2b01 = [i for i in ab_ab_tmp if "2" in i][0]
	a, b = get_ab(formulas)
	content = textwrap.indent(textwrap.dedent(f"""\
		case Label::ab_ab::{ab_ab}:
		{{
			const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
				list_Aa01_Da,
				data_wrapper(Label::ab::{a01b01}).Ds_ab );
			const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
				list_Aa2_Da,
				data_wrapper(Label::ab::{a2b01}).Ds_ab );
			const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set( LRI_Cal_Aux::filter_list_set(
				list_Ab01_Db,
				data_wrapper(Label::ab::{a01b01}).index_Ds_ab[0]),
				data_wrapper(Label::ab::{a2b01}).index_Ds_ab[0]);
			const std::vector<TAC> &list_Ab2 =
				list_Ab2_Db;

			for(const TA &Aa01 : list_Aa01)
			{{
				if(this->filter_atom->filter_for1(label,Aa01))	continue;
				std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

				#pragma omp for schedule(dynamic) nowait
				for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
				{{
					const TAC &Ab01 = list_Ab01[ib01];
					if(this->filter_atom->filter_for2(label,Aa01,Ab01))	continue;
					const Tensor<Tdata> &D_{a01b01} = tools.get_Ds_ab(Label::ab::{a01b01}, Aa01, Ab01);
					if(D_{a01b01}.empty())	continue;
					// D_mul = D_a * D_{a2b01}
					Tensor<Tdata> D_mul;
					for(const TAC &Aa2 : list_Aa2)
					{{
						if(this->filter_atom->filter_for31(label,Aa01,Ab01,Aa2))	continue;
						const Tensor<Tdata> &D_a = {get_D_a(a)};
						if(D_a.empty())	continue;
						const Tensor<Tdata> &D_{a2b01} = tools.get_Ds_ab(Label::ab::{a2b01}, Aa2, Ab01);
						if(D_{a2b01}.empty())	continue;

						// {formulas[0]}
						Tensor<Tdata> D_tmp1 = Tensor_Multiply::{get_call(formulas[0],None)};
						LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
					}}
					if(D_mul.empty())	continue;

					// D_result = D_mul * D_{a01b01} * D_b
					for(const TAC &Ab2 : list_Ab2)
					{{
						if(this->filter_atom->filter_for32(label,Aa01,Ab01,Ab2))	continue;
						const Tensor<Tdata> &D_b = {get_D_b(b)};
						if(D_b.empty())	continue;

						// {formulas[1]}
						const Tensor<Tdata> D_tmp2 = Tensor_Multiply::{get_call(formulas[1],"mul")};
						// {formulas[2]}
						Tensor<Tdata> D_tmp3 = Tensor_Multiply::{get_call(formulas[2],"tmp2")};
						LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab2]);
					}}
				}} // end for Ab01

				if(!Ds_result_fixed.empty())
					LRI_Cal_Aux::add_Ds(std::move(Ds_result_fixed),
										Ds_result_thread[Aa01]);
				LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
			}} // end for Aa01
		}} break; // end case {ab_ab}
		"""),
		"\t"*4)
	return content



def content_a01b01_a2b2(ab_ab, formulas):
	ab_ab_tmp = ab_ab.split("_")
	a01b01 = [i for i in ab_ab_tmp if not "2" in i][0]
	a, b = get_ab(formulas)
	content = textwrap.indent(textwrap.dedent(f"""\
		case Label::ab_ab::{ab_ab}:
		{{
			const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
				list_Aa01_Da,
				data_wrapper(Label::ab::{a01b01}).Ds_ab );
			const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
				list_Aa2_Da,
				data_wrapper(Label::ab::a2b2).Ds_ab );
			const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
				list_Ab01_Db,
				data_wrapper(Label::ab::{a01b01}).index_Ds_ab[0]);
			const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
				list_Ab2_Db,
				data_wrapper(Label::ab::a2b2).index_Ds_ab[0]);

			for(const TA &Aa01 : list_Aa01)
			{{
				if(this->filter_atom->filter_for1(label,Aa01))	continue;
				std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

				#pragma omp for schedule(dynamic) nowait
				for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
				{{
					const TAC &Ab2 = list_Ab2[ib2];
					if(this->filter_atom->filter_for2(label,Aa01,Ab2))	continue;
					// D_mul = D_a * D_a2b2
					Tensor<Tdata> D_mul;
					for(const TAC &Aa2 : list_Aa2)
					{{
						if(this->filter_atom->filter_for31(label,Aa01,Ab2,Aa2))	continue;
						const Tensor<Tdata> &D_a = {get_D_a(a)};
						if(D_a.empty())	continue;
						const Tensor<Tdata> &D_a2b2 = tools.get_Ds_ab(Label::ab::a2b2, Aa2, Ab2);
						if(D_a2b2.empty())	continue;

						// {formulas[0]}
						Tensor<Tdata> D_tmp1 = Tensor_Multiply::{get_call(formulas[0],None)};
						LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
					}}
					if(D_mul.empty())	continue;

					// D_result = D_mul * D_{a01b01} * D_b
					for(const TAC &Ab01 : list_Ab01)
					{{
						if(this->filter_atom->filter_for32(label,Aa01,Ab2,Ab01))	continue;
						const Tensor<Tdata> &D_b = {get_D_b(b)};
						if(D_b.empty())	continue;
						const Tensor<Tdata> &D_{a01b01} = tools.get_Ds_ab(Label::ab::{a01b01}, Aa01, Ab01);
						if(D_{a01b01}.empty())	continue;

						// {formulas[1]}
						const Tensor<Tdata> D_tmp2 = Tensor_Multiply::{get_call(formulas[1],"mul")};
						// {formulas[2]}
						Tensor<Tdata> D_tmp3 = Tensor_Multiply::{get_call(formulas[2],"tmp2")};
						LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab01]);
					}}
				}} // end for Ab01

				if(!Ds_result_fixed.empty())
					LRI_Cal_Aux::add_Ds(std::move(Ds_result_fixed),
										Ds_result_thread[Aa01]);
				LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
			}} // end for Aa01
		}} break; // end case {ab_ab}
		"""),
		"\t"*4)
	return content



def content_a01b2_a2b01(ab_ab, formulas):
	ab_ab_tmp = ab_ab.split("_")
	a01b2 = [i for i in ab_ab_tmp if "b2" in i][0]
	a2b01 = [i for i in ab_ab_tmp if "a2" in i][0]
	a, b = get_ab(formulas)
	content = textwrap.indent(textwrap.dedent(f"""\
		case Label::ab_ab::{ab_ab}:
		{{
			const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
				list_Aa01_Da,
				data_wrapper(Label::ab::{a01b2}).Ds_ab );
			const std::vector<TAC> &list_Aa2 = LRI_Cal_Aux::filter_list_map(
				list_Aa2_Da,
				data_wrapper(Label::ab::{a2b01}).Ds_ab );
			const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
				list_Ab01_Db,
				data_wrapper(Label::ab::{a2b01}).index_Ds_ab[0]);
			const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
				list_Ab2_Db,
				data_wrapper(Label::ab::{a01b2}).index_Ds_ab[0]);

			for(const TA &Aa01 : list_Aa01)
			{{
				if(this->filter_atom->filter_for1(label,Aa01))	continue;
				#pragma omp for schedule(dynamic) nowait
				for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
				{{
					const TAC &Ab01 = list_Ab01[ib01];
					if(this->filter_atom->filter_for2(label,Aa01,Ab01))	continue;
					// D_mul1 = D_b * D_{a01b2}
					Tensor<Tdata> D_mul1;
					for(const TAC &Ab2 : list_Ab2)
					{{
						if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
						const Tensor<Tdata> &D_b = {get_D_b(b)};
						if(D_b.empty())	continue;
						const Tensor<Tdata> &D_{a01b2} = tools.get_Ds_ab(Label::ab::{a01b2}, Aa01, Ab2);
						if(D_{a01b2}.empty())	continue;

						// {formulas[0]}
						Tensor<Tdata> D_tmp1 = Tensor_Multiply::{get_call(formulas[0],None)};
						LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul1);
					}}
					if(D_mul1.empty())	continue;

					// D_mul2 = D_{a2b01} * D_a
					Tensor<Tdata> D_mul2;
					for(const TAC &Aa2 : list_Aa2)
					{{
						if(this->filter_atom->filter_for32(label,Aa01,Ab01,Aa2))	continue;
						const Tensor<Tdata> &D_a = {get_D_a(a)};
						if(D_a.empty())	continue;
						const Tensor<Tdata> &D_{a2b01} = tools.get_Ds_ab(Label::ab::{a2b01}, Aa2, Ab01);
						if(D_{a2b01}.empty())	continue;
						// {formulas[1]}
						Tensor<Tdata> D_tmp2 = Tensor_Multiply::{get_call(formulas[1],None)};
						LRI_Cal_Aux::add_Ds(std::move(D_tmp2), D_mul2);
					}}
					if(D_mul2.empty())	continue;

					// D_result = D_mul2 * D_mul1
					// {formulas[2]}
					Tensor<Tdata> D_mul3 = Tensor_Multiply::{get_func(formulas[2])}(D_mul2, D_mul1);
					LRI_Cal_Aux::add_Ds(std::move(D_mul3),
										Ds_result_thread[Aa01][Ab01]);
				}} // end for Ab01

				LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
			}} // end for Aa01
		}} break; // end case {ab_ab}
		"""),
		"\t"*4)
	return content



def content_prefix():
	content = textwrap.dedent("""\
		// ===================
		//  Author: Peize Lin
		//  date: 2023.08.02
		// ===================

		// This file is automatically generated from tools/cal_loop3/cal_loop3_CPU.py

		#pragma once

		#include "LRI.h"
		#include "LRI_Cal_Aux.h"
		#include "../global/Array_Operator.h"
		#include "../global/Tensor_Multiply.h"

		#include <omp.h>
		#ifdef __MKL_RI
		#include <mkl_service.h>
		#endif

		namespace RI
		{

		template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
		void LRI<TA,Tcell,Ndim,Tdata>::cal_loop3_CPU(
			const std::vector<Label::ab_ab> &labels,
			std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_result,
			const Tdata fac_add_Ds)
		{
			using namespace Array_Operator;

			const Data_Pack_Wrapper<TA,TC,Tdata> data_wrapper(this->data_pool, this->data_ab_name);
			const LRI_Cal_Tools<TA,TC,Tdata> tools(this->period, this->data_pool, this->data_ab_name);

			std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_a_transpose, Ds_b_transpose;
			std::tie(Ds_a_transpose, Ds_b_transpose) = tools.cal_Ds_transpose(labels);

		  #ifdef __MKL_RI
			const std::size_t mkl_threads = mkl_get_max_threads();
			mkl_set_num_threads(1);
		  #endif

			std::map<TA, omp_lock_t> lock_Ds_result_add_map = LRI_Cal_Aux::init_lock_result(labels, this->parallel->list_A, Ds_result);

			#pragma omp parallel
			{
				std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_result_thread;

				for(const Label::ab_ab &label : labels)
				{
					const std::vector<TA>  list_Aa01_Da = LRI_Cal_Aux::filter_list_map( this->parallel->list_A.at(Label_Tools::to_Aab_Aab(label)).a01, data_wrapper(Label::ab::a).Ds_ab );
					const std::vector<TAC> list_Ab01_Db = LRI_Cal_Aux::filter_list_map( this->parallel->list_A.at(Label_Tools::to_Aab_Aab(label)).b01, data_wrapper(Label::ab::b).Ds_ab );
					const std::vector<TAC> list_Aa2_Da  = LRI_Cal_Aux::filter_list_set( this->parallel->list_A.at(Label_Tools::to_Aab_Aab(label)).a2,  data_wrapper(Label::ab::a).index_Ds_ab[0] );
					const std::vector<TAC> list_Ab2_Db  = LRI_Cal_Aux::filter_list_set( this->parallel->list_A.at(Label_Tools::to_Aab_Aab(label)).b2,  data_wrapper(Label::ab::b).index_Ds_ab[0] );
					switch(label)
					{
		""")
	return content




def content_suffix():
	content = textwrap.dedent("""\
						default:
							throw std::invalid_argument(std::string(__FILE__)+std::to_string(__LINE__));
					} // end switch(label)
				} // end for label

				LRI_Cal_Aux::add_Ds_omp_wait_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
			} // end #pragma omp parallel

			LRI_Cal_Aux::destroy_lock(lock_Ds_result_add_map);
			LRI_Cal_Aux::destroy_Ds_result(Ds_result);

		  #ifdef __MKL_RI
			mkl_set_num_threads(mkl_threads);
		  #endif
		}	// end LRI::cal_loop3()

		}	// end namespace RI
		""")
	return content




def print_all():

	a01b01_a01b01 = {
		"a0b0_a1b1":
			["a1a2b0 = a0a1a2 * a0b0",
			"a2b0b1 = a1a2b0 * a1b1",
			"a2b2 = a2b0b1 * b0b1b2"],
		"a0b1_a1b0":
			["a0a2b0 = a1a0a2 * a1b0",
			"a2b0b1 = a0a2b0 * a0b1",
			"a2b2 = a2b0b1 * b0b1b2"]}

	a01b01_a01b2 = {
		"a0b0_a1b2":
			["b0b1a1 = b0b1b2 * a1b2",
			"b1a1a0 = b0b1a1 * a0b0",
			"a2b1 = a1a0a2 * b1a1a0"],
		"a0b1_a1b2":
			["a1b0b1 = a1b2 * b0b1b2",
			"a0a1b0 = a0b1 * a1b0b1",
			"a2b0 = a0a1a2 * a0a1b0"],
		"a0b2_a1b0":
			["b0b1a0 = b0b1b2 * a0b2",
			"b1a0a1 = b0b1a0 * a1b0",
			"a2b1 = a0a1a2 * b1a0a1"],
		"a0b2_a1b1":
			["a0b0b1 = a0b2 * b0b1b2",
			"a1a0b0 = a1b1 * a0b0b1",
			"a2b0 = a1a0a2 * a1a0b0"]}

	a01b01_a2b01 = {
		"a0b0_a2b1":
			["b1a1a0 = a2b1 * a1a0a2",
			"b0b1a1 = a0b0 * b1a1a0",
			"a1b2 = b0b1a1 * b0b1b2"],
		"a0b1_a2b0":
			["a0a1b0 = a0a1a2 * a2b0",
			"a1b0b1 = a0a1b0 * a0b1",
			"a1b2 = a1b0b1 * b0b1b2"],
		"a1b0_a2b1":
			["b1a0a1 = a2b1 * a0a1a2",
			"b0b1a0 = a1b0 * b1a0a1",
			"a0b2 = b0b1a0 * b0b1b2"],
		"a1b1_a2b0":
			["a1a0b0 = a1a0a2 * a2b0",
			"a0b0b1 = a1a0b0 * a1b1",
			"a0b2 = a0b0b1 * b0b1b2"]}

	a01b01_a2b2 = {
		"a0b0_a2b2":
			["b2a1a0 = a2b2 * a1a0a2",
			"b0b2a1 = a0b0 * b2a1a0",
			"a1b1 = b0b2a1 * b1b0b2"],
		"a0b1_a2b2":
			["b2a1a0 = a2b2 * a1a0a2",
			"b1b2a1 = a0b1 * a2a1a0",
			"a1b0 = b1b2a1 * b0b1b2"],
		"a1b0_a2b2":
			["b2a0a1 = a2b2 * a0a1a2",
			"b0b2a0 = a1b0 * b2a0a1",
			"a0b1 = b0b2a0 * b1b0b2"],
		"a1b1_a2b2":
			["b2a0a1 = a2b2 * a0a1a2",
			"b1b2a0 = a1b1 * b2a0a1",
			"a0b0 = b1b2a0 * b0b1b2"]}

	a01b2_a2b01 = {
		"a1b2_a2b1":
			["b0b1a1 = b0b1b2 * a1b2",
			"b1a1a0 = a2b1 * a1a0a2",
			"a0b0 = b1a1a0 * b0b1a1"],
		"a0b2_a2b0":
			["a0b0b1 = a0b2 * b0b1b2",
			"a1a0b0 = a1a0a2 * a2b0",
			"b1a1 = a1a0b0 * a0b0b1"],
		"a0b2_a2b1":
			["b0b1a0 = b0b1b2 * a0b2",
			"b1a0a1 = a2b1 * a0a1a2",
			"a1b0 = b1a0a1 * b0b1a0"],
		"a1b2_a2b0":
			["a1b0b1 = a1b2 * b0b1b2",
			"a0a1b0 = a0a1a2 * a2b0",
			"a0b1 = a0a1b0 * a1b0b1"]}
	

	with open("../../include/RI/ri/LRI-cal_loop3_CPU.hpp", "w", newline="\r\n") as file:
		print(content_prefix(), file=file)
		for Aab_Aab in ["a01b01_a01b01", "a01b01_a01b2", "a01b01_a2b01", "a01b01_a2b2", "a01b2_a2b01"]:
			print(f"			  // Aab_Aab::{Aab_Aab}\n", file=file)
			for ab_ab, formulas in eval(Aab_Aab).items():
				print(eval("content_"+Aab_Aab)(ab_ab,formulas), file=file)
		print(content_suffix(), file=file)


print_all()
