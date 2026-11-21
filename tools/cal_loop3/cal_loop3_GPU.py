# -*- coding: utf-8 -*-
# =================================
#  Author: Peize Lin, Laiyuan Yang
#  date: 2026.01.01
# =================================

import textwrap
import itertools

get_D_a = lambda a : "tools.get_Ds_ab(Label::ab::a, Aa01, Aa2)" \
					if a=="a0a1a2" else "Global_Func::find(Ds_a_transpose, Aa01, Aa2)"
get_D_b = lambda b : "tools.get_Ds_ab(Label::ab::b, Ab01, Ab2)" \
					if b=="b0b1b2" else "Global_Func::find(Ds_b_transpose, Ab01.first, TAC{Ab2.first, (Ab2.second-Ab01.second)%this->period})"

ab_list = ["a0a1a2", "a1a0a2", "b0b1b2", "b1b0b2"] + \
	[a+b for a,b in itertools.product(["a0","a1","a2"], ["b0","b1","b2"])]

def get_ab(formulas):
	for formula in formulas:
		prod_name, mulx_name, muly_name = formula.split()[::2]
		for D in (mulx_name, muly_name):
			if D in ["a0a1a2", "a1a0a2"]:
				a = D
			elif D in ["b0b1b2", "b1b0b2"]:
				b = D
	return a,b


def get_gemm_param(formula, *, other=""):
	""" formula = "a1a2b0 = a0a1a2 * a0b0" """

	def split_tensor(lst):
		return [lst[i : i + 2] for i in range(0, len(lst), 2)]

	# 2, 1, 4 = get_equal([x0,x1,a,b,c,d,x6,...], [y0,a,b,c,d,y5,...])
	def get_equal(mulx_split, muly_split):
		for ix,ux in enumerate(mulx_split):
			for iy,uy in enumerate(muly_split):
				if ux==uy:
					for i in range(1,100):
						if ix+i==len(mulx_split) or iy+i==len(muly_split) or mulx_split[ix+i]!=muly_split[iy+i]:
							return ix,iy,i
		raise AttributeError(f"{mulx_split}, {muly_split}")

	prod, mulx, muly = formula.split()[::2]
	def split_formula(s):
		return [s[i:i+2] for i in range(0, len(s), 2)]
	mulx_split = split_formula(mulx)
	muly_split = split_formula(muly)
	index_x, index_y, index_len = get_equal(mulx_split, muly_split)
	Tx = "GPU_Backend::Trans" if index_x==0 else "GPU_Backend::NoTrans"
	Ty = "GPU_Backend::Trans" if index_y!=0 else "GPU_Backend::NoTrans"
	T_all = Tx, Ty

	def get_dim(ab, index, other):
		def get_D():
			if len(split_tensor(ab)) == 2:
				return ab
			elif ab in ["a0a1a2", "a1a0a2"]:
				return "a"
			elif ab in ["b0b1b2", "b1b0b2"]:
				return "b"
			else:
				return other
		D = f"D_{get_D()}"
		return ", ".join([f"{D}.shape[{i}]" for i in index])

	def add_brace(dim):
		return f"{{{dim}}}" if "," in dim else dim

	if type(other)==str:
		other = [other]*2
	if index_x==0:
		dim_m = get_dim(mulx, list(range(index_x+index_len, len(mulx_split))), other[0])
	else:
		dim_m = get_dim(mulx, list(range(0, index_x)), other[0])
	if index_y==0:
		dim_n = get_dim(muly, list(range(index_y+index_len, len(muly_split))), other[1])
	else:
		dim_n = get_dim(muly, list(range(0, index_y)), other[1])
	if mulx in ab_list:
		dim_k = get_dim(mulx, list(range(index_x, index_x+index_len)), other[0])
	else:
		dim_k = get_dim(muly, list(range(index_y, index_y+index_len)), other[1])
	dim_all = add_brace(dim_m), add_brace(dim_n), add_brace(dim_k), dim_m, dim_n

	def get_d_array(ab, other_one):
		if len(split_tensor(ab)) == 2:
			return f"rDs_{ab}.d_array"
		elif ab in ["a0a1a2", "a1a0a2"]:
			return "rDs_a.d_array"
		elif ab in ["b0b1b2", "b1b0b2"]:
			return "rDs_b.d_array"
		else:
			if "mul" in other_one:
				return f"rDs_{other_one}.d_array_2"
			elif other_one=="tmp":
				return f""
			else:
				return f"rDs_{other_one}.d_array"

	d_array_x = get_d_array(mulx, other[0])
	d_array_y = get_d_array(muly, other[1])
	if not d_array_x:
		d_array_all = d_array_y, ""
	elif not d_array_y:
		d_array_all = d_array_x, ""
	else:
		d_array_all = d_array_x, d_array_y

	return T_all + dim_all + d_array_all


def get_C0_left(formula_0, formula_1):
	prod_0, mulx_0, muly_0 = formula_0.split()[::2]
	prod_1, mulx_1, muly_1 = formula_1.split()[::2]
	if prod_0 == mulx_1:
		return "true"
	else:
		return "false"



def content_a01b01_a01b01(ab_ab, formulas):
	a01b01 = ab_ab.split("_")
	a, b = get_ab(formulas)
	Tx_0, Ty_0, dim_m_0_brace, dim_n_0_brace, dim_k_0_brace, dim_m_0_raw, dim_n_0_raw, rDs_x_0, rDs_y_0 = get_gemm_param(formulas[0])
	Tx_1, Ty_1, dim_m_1_brace, dim_n_1_brace, dim_k_1_brace, dim_m_1_raw, dim_n_1_raw, rDs_x_1, rDs_y_1 = get_gemm_param(formulas[1], other="tmp")
	Tx_2, Ty_2, dim_m_2_brace, dim_n_2_brace, dim_k_2_brace, dim_m_2_raw, dim_n_2_raw, rDs_x_2, rDs_y_2 = get_gemm_param(formulas[2], other="mul")

	content = textwrap.indent(
		textwrap.dedent(
			f"""\
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

			GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_{a01b01[0]}, rDs_{a01b01[1]};
			GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
			GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
			GPU_Data::Output<TA, TAC, Tdata> rDs_output;

			Dim_mnk dim_0, dim_1, dim_2;

			// D_mul = D_b * D_{a01b01[0]} * D_{a01b01[1]}
			#pragma omp parallel
			{{
				for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
				{{
					const TA &Aa01 = list_Aa01[ia01];
					if (this->filter_atom->filter_for1(label, Aa01))	continue;
					#pragma omp for schedule(dynamic) nowait
					for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01)
					{{
						const TAC &Ab01 = list_Ab01[ib01];
						if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
						const Tensor<Tdata> &D_{a01b01[0]} = tools.get_Ds_ab(Label::ab::{a01b01[0]}, Aa01, Ab01);
						if(D_{a01b01[0]}.empty())	continue;
						const Tensor<Tdata> &D_{a01b01[1]} = tools.get_Ds_ab(Label::ab::{a01b01[1]}, Aa01, Ab01);
						if(D_{a01b01[1]}.empty())	continue;
						for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2)
						{{
							const TAC &Ab2 = list_Ab2[ib2];
							if (this->filter_atom->filter_for31(label, Aa01, Ab01, Ab2))	continue;
							const Tensor<Tdata> &D_b = {get_D_b(b)};
							if(D_b.empty())	continue;

							#pragma omp critical(rDs_mul)
							{{
								rDs_b.insert(Ab01.first, {{Ab2.first, (Ab2.second - Ab01.second) % period}}, D_b);
								rDs_{a01b01[0]}.insert(Aa01, Ab01, D_{a01b01[0]});
								rDs_{a01b01[1]}.insert(Aa01, Ab01, D_{a01b01[1]});
								const GPU_Data::Pack &D_tmp = rDs_tmp.insert({{{dim_m_0_raw}, {dim_n_0_raw}}});
								const GPU_Data::Pack &D_mul = rDs_mul.insert_1st(Aa01, Ab2, {{{dim_m_1_raw}, {dim_n_1_raw}}});
								dim_0.input({dim_m_0_brace}, {dim_n_0_brace}, {dim_k_0_brace});
								dim_1.input({dim_m_1_brace}, {dim_n_1_brace}, {dim_k_1_brace});
							}}
							rDs_b.insert_data(D_b);
							rDs_{a01b01[0]}.insert_data(D_{a01b01[0]});
							rDs_{a01b01[1]}.insert_data(D_{a01b01[1]});
						}} // end for Aa01
					}} // end for Ab2
				}} // end for Ab01
			}} // end omp parallel

			rDs_b.upload(queue);
			rDs_{a01b01[0]}.upload(queue);
			rDs_{a01b01[1]}.upload(queue);
			const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
			rDs_mul.upload_1st(queue);
			dim_0.upload(queue);
			dim_1.upload(queue);

			constexpr bool C0_left = {get_C0_left(formulas[0], formulas[1])};
			GPU_Backend::gemmVbatched2s(
				{Tx_0}, {Ty_0},
				dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_0}, {rDs_y_0},
				GPU_Backend::Type_to_GPU<Tdata>{{0}}, rDs_tmp.d_array,
				{Tx_1}, {Ty_1},
				dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_1},
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, rDs_mul.d_array_1,
				C0_left,
				rDs_mul.h_array_1.size(), rDs_tmp_segments_size, queue);
			GPU_Backend::sync(queue);

			// D_result = D_mul * D_a
			GPU_Data::Input<TA, TAC, Tdata> rDs_a;
			#pragma omp parallel
			{{
				for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
				{{
					const TA &Aa01 = list_Aa01[ia01];
					if (this->filter_atom->filter_for1(label, Aa01))	continue;
					#pragma omp for schedule(dynamic) nowait
					for (std::size_t ia2 = 0; ia2 < list_Aa2.size(); ++ia2) // F
					{{
						const TAC &Aa2 = list_Aa2[ia2];
						if (this->filter_atom->filter_for2(label, Aa01, Aa2))	continue;
						const Tensor<Tdata> &D_a = {get_D_a(a)};
						if (D_a.empty())	continue;
						for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2) // G
						{{
							const TAC &Ab2 = list_Ab2[ib2];
							if (this->filter_atom->filter_for32(label, Aa01, Aa2, Ab2))	continue;
							const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab2);
							if (!D_mul.exist)	continue;

							#pragma omp critical(rDs_output)
							{{
								rDs_mul.insert_2nd(D_mul);
								rDs_a.insert(Aa01, Aa2, D_a);
								rDs_output.insert( Aa2.first, {{Ab2.first, (Ab2.second - Aa2.second) % period}}, {{{dim_m_2_raw}, {dim_n_2_raw}}});
								dim_2.input({dim_m_2_brace}, {dim_n_2_brace}, {dim_k_2_brace});
							}}
							rDs_a.insert_data(D_a);
						}} // end for Ab2
					}} // end for Aa2
				}} // end for Aa01
			}} // end omp parallel

			rDs_output.upload(queue);
			rDs_mul.upload_2nd(queue);
			rDs_a.upload(queue);
			dim_2.upload(queue);

			GPU_Backend::gemmVbatched(
				{Tx_2}, {Ty_2},
				dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_2}, {rDs_y_2},
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, rDs_output.d_array,
				rDs_output.h_array.size(), queue);
			GPU_Backend::sync(queue);

			rDs_output.download(Ds_result, queue, fac_add_Ds);
		}} break; // end case {ab_ab}
		"""
		),
		"\t" * 4,
	)
	return content


def content_a01b01_a01b2(ab_ab, formulas):
	ab_ab_tmp = ab_ab.split("_")
	a01b01 = [i for i in ab_ab_tmp if not "2" in i][0]
	a01b2 = [i for i in ab_ab_tmp if "2" in i][0]
	a, b = get_ab(formulas)
	Tx_0, Ty_0, dim_m_0_brace, dim_n_0_brace, dim_k_0_brace, dim_m_0_raw, dim_n_0_raw, rDs_x_0, rDs_y_0 = get_gemm_param(formulas[0])
	Tx_1, Ty_1, dim_m_1_brace, dim_n_1_brace, dim_k_1_brace, dim_m_1_raw, dim_n_1_raw, rDs_x_1, rDs_y_1 = get_gemm_param(formulas[1], other="mul")
	Tx_2, Ty_2, dim_m_2_brace, dim_n_2_brace, dim_k_2_brace, dim_m_2_raw, dim_n_2_raw, rDs_x_2, rDs_y_2 = get_gemm_param(formulas[2], other="tmp")

	content = textwrap.indent(
		textwrap.dedent(
			f"""\
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

			GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_{a01b2};
			GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
			GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
			GPU_Data::Output<TA, TAC, Tdata> rDs_output;

			Dim_mnk dim_0, dim_1, dim_2;

			// {formulas[0]}
			// D_mul = D_b * D_{a01b2}
			#pragma omp parallel
			{{
				for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
				{{
					const TAC &Ab01 = list_Ab01[ib01];
					if(this->filter_atom->filter_for1(label,Ab01))	continue;
					#pragma omp for schedule(dynamic) nowait
					for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
					{{
						const TAC &Ab2 = list_Ab2[ib2];
						if(this->filter_atom->filter_for2(label,Ab01,Ab2))	continue;
						const Tensor<Tdata> &D_b = {get_D_b(b)};
						if(D_b.empty())	continue;
						for (std::size_t ia01= 0; ia01<list_Aa01.size(); ++ia01)
						{{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
							const Tensor<Tdata> &D_{a01b2} = tools.get_Ds_ab(Label::ab::{a01b2}, Aa01, Ab2);
							if(D_{a01b2}.empty())	continue;

							#pragma omp critical(rDs_mul)
							{{
								rDs_b.insert(Ab01.first, {{Ab2.first, (Ab2.second - Ab01.second) % period}}, D_b);
								rDs_{a01b2}.insert(Aa01, Ab2, D_{a01b2});
								const GPU_Data::Pack &D_mul = rDs_mul.insert_1st(Aa01, Ab01, {{{dim_m_0_raw}, {dim_n_0_raw}}});
								dim_0.input({dim_m_0_brace}, {dim_n_0_brace}, {dim_k_0_brace});
							}}
							rDs_b.insert_data(D_b);
							rDs_{a01b2}.insert_data(D_{a01b2});
						}} // end for Ab2
					}} // end for Aa01
				}} // end for Ab01
			}} // end omp parallel

			rDs_b.upload(queue);
			rDs_{a01b2}.upload(queue);
			rDs_mul.upload_1st(queue);

			dim_0.upload(queue);

			GPU_Backend::gemmVbatched(
				{Tx_0}, {Ty_0},
				dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_0}, {rDs_y_0},
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, rDs_mul.d_array_1,
				rDs_mul.h_array_1.size(), queue);
			GPU_Backend::sync(queue);

			// D_result = D_mul * D_a * D_{a01b01}
			GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_{a01b01};

			#pragma omp parallel
			{{
				for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
				{{
					const TA &Aa01 = list_Aa01[ia01];
					if (this->filter_atom->filter_for1(label, Aa01))	continue;
					#pragma omp for schedule(dynamic) nowait
					for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01) // G
					{{
						const TAC &Ab01 = list_Ab01[ib01];
						if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
						const Tensor<Tdata> &D_{a01b01} = tools.get_Ds_ab(Label::ab::{a01b01}, Aa01, Ab01);
						if(D_{a01b01}.empty())	continue;
						const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab01);
						if (!D_mul.exist)	continue;
						for (std::size_t ia2 = 0; ia2 < list_Aa2.size(); ++ia2) // F
						{{
							const TAC &Aa2 = list_Aa2[ia2];
							if (this->filter_atom->filter_for32(label, Aa01, Aa2, Ab01))	continue;
							const Tensor<Tdata> &D_a = {get_D_a(a)};
							if (D_a.empty())	continue;

							#pragma omp critical(rDs_output)
							{{
								rDs_a.insert(Aa01, Aa2, D_a);
								rDs_{a01b01}.insert(Aa01, Ab01, D_{a01b01});
								rDs_mul.insert_2nd(D_mul);
								const GPU_Data::Pack &D_tmp = rDs_tmp.insert({{{dim_m_1_raw}, {dim_n_1_raw}}});
								rDs_output.insert(Aa2.first, {{Ab01.first, (Ab01.second - Aa2.second) % period}}, {{{dim_m_2_raw}, {dim_n_2_raw}}});
								dim_1.input({dim_m_1_brace}, {dim_n_1_brace}, {dim_k_1_brace});
								dim_2.input({dim_m_2_brace}, {dim_n_2_brace}, {dim_k_2_brace});
							}}
							rDs_a.insert_data(D_a);
							rDs_{a01b01}.insert_data(D_{a01b01});
						}} // end for Ab01
					}}  // end for Aa2
				}} // end for Aa01
			}} // end omp parallel

			const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
			rDs_output.upload(queue);
			rDs_mul.upload_2nd(queue);
			rDs_a.upload(queue);
			rDs_{a01b01}.upload(queue);

			dim_1.upload(queue);
			dim_2.upload(queue);

			constexpr bool C0_left = {get_C0_left(formulas[1], formulas[2])};
			GPU_Backend::gemmVbatched2s(
				{Tx_1}, {Ty_1},
				dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_1}, {rDs_y_1},
				GPU_Backend::Type_to_GPU<Tdata>{{0}}, rDs_tmp.d_array,
				{Tx_2}, {Ty_2},
				dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_2},
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, rDs_output.d_array,
				C0_left,
				rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
			GPU_Backend::sync(queue);

			rDs_output.download(Ds_result, queue, fac_add_Ds);
		}} break; // end case {ab_ab}
		"""
		),
		"\t" * 4,
	)
	return content


def content_a01b01_a2b01(ab_ab, formulas):
	ab_ab_tmp = ab_ab.split("_")
	a01b01 = [i for i in ab_ab_tmp if not "2" in i][0]
	a2b01 = [i for i in ab_ab_tmp if "2" in i][0]
	a, b = get_ab(formulas)
	Tx_0, Ty_0, dim_m_0_brace, dim_n_0_brace, dim_k_0_brace, dim_m_0_raw, dim_n_0_raw, rDs_x_0, rDs_y_0 = get_gemm_param(formulas[0])
	Tx_1, Ty_1, dim_m_1_brace, dim_n_1_brace, dim_k_1_brace, dim_m_1_raw, dim_n_1_raw, rDs_x_1, rDs_y_1 = get_gemm_param(formulas[1], other="mul")
	Tx_2, Ty_2, dim_m_2_brace, dim_n_2_brace, dim_k_2_brace, dim_m_2_raw, dim_n_2_raw, rDs_x_2, rDs_y_2 = get_gemm_param(formulas[2], other="tmp")

	content = textwrap.indent(
		textwrap.dedent(
			f"""\
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

			GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_{a2b01};
			GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
			GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
			GPU_Data::Output<TA, TAC, Tdata> rDs_output;

			Dim_mnk dim_0, dim_1, dim_2;

			// {formulas[0]}
			// D_mul = D_a * D_{a2b01}
			#pragma omp parallel
			{{
				for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
				{{
					const TA &Aa01 = list_Aa01[ia01];
					if(this->filter_atom->filter_for1(label,Aa01))	continue;
					#pragma omp for schedule(dynamic) nowait
					for(std::size_t ia2=0; ia2<list_Aa2.size(); ++ia2)
					{{
						const TAC &Aa2 = list_Aa2[ia2];
						if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
						const Tensor<Tdata> &D_a = {get_D_a(a)};
						if(D_a.empty())	continue;
						for (std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for31(label,Aa01,Ab01,Aa2))	continue;
							const Tensor<Tdata> &D_{a2b01} = tools.get_Ds_ab(Label::ab::{a2b01}, Aa2, Ab01);
							if(D_{a2b01}.empty())	continue;

							#pragma omp critical(rDs_mul)
							{{
								rDs_a.insert(Aa01, Aa2, D_a);
								rDs_{a2b01}.insert(Aa2.first, {{Ab01.first, (Ab01.second - Aa2.second) % this->period}}, D_{a2b01});
								const GPU_Data::Pack &D_mul = rDs_mul.insert_1st(Aa01, Ab01, {{{dim_m_0_raw}, {dim_n_0_raw}}});
								dim_0.input({dim_m_0_brace}, {dim_n_0_brace}, {dim_k_0_brace});
							}}
							rDs_a.insert_data(D_a);
							rDs_{a2b01}.insert_data(D_{a2b01});

						}} // end for Aa2
					}} // end for Ab01
				}} // end for Aa01
			}} // end omp parallel

			rDs_a.upload(queue);
			rDs_{a2b01}.upload(queue);
			rDs_mul.upload_1st(queue);
			dim_0.upload(queue);

			GPU_Backend::gemmVbatched(
				{Tx_0}, {Ty_0},
				dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_0}, {rDs_y_0},
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, rDs_mul.d_array_1,
				rDs_mul.h_array_1.size(), queue);
			GPU_Backend::sync(queue);

			// D_result = D_mul * D_{a01b01} * D_b
			GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_{a01b01};

			#pragma omp parallel
			{{
				for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01) // A
				{{
					const TA &Aa01 = list_Aa01[ia01];
					if (this->filter_atom->filter_for1(label, Aa01))	continue;
					#pragma omp for schedule(dynamic) nowait
					for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01) // G
					{{
						const TAC &Ab01 = list_Ab01[ib01];
						if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
						const Tensor<Tdata> &D_{a01b01} = tools.get_Ds_ab(Label::ab::{a01b01}, Aa01, Ab01);
						if(D_{a01b01}.empty())	continue;
						const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab01);
						if (!D_mul.exist)	continue;
						for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2)
						{{
							const TAC &Ab2 = list_Ab2[ib2];
							if (this->filter_atom->filter_for32(label, Aa01, Ab01, Ab2))	continue;
							const Tensor<Tdata> &D_b = {get_D_b(b)};
							if(D_b.empty())	continue;

							#pragma omp critical(rDs_output)
							{{
								rDs_b.insert(Ab01.first, {{Ab2.first, (Ab2.second - Ab01.second) % period}}, D_b);
								rDs_{a01b01}.insert(Aa01, Ab01, D_{a01b01});
								rDs_mul.insert_2nd(D_mul);
								const GPU_Data::Pack &D_tmp = rDs_tmp.insert({{{dim_m_1_raw}, {dim_n_1_raw}}});
								rDs_output.insert(Aa01, Ab2, {{{dim_m_2_raw}, {dim_n_2_raw}}});
								dim_1.input({dim_m_1_brace}, {dim_n_1_brace}, {dim_k_1_brace});
								dim_2.input({dim_m_2_brace}, {dim_n_2_brace}, {dim_k_2_brace});
							}}
							rDs_b.insert_data(D_b);
							rDs_{a01b01}.insert_data(D_{a01b01});
						}} // end for Aa01
					}} // end for Ab2
				}} // end for Ab01
			}} // end omp parallel

			const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
			rDs_output.upload(queue);
			rDs_mul.upload_2nd(queue);
			rDs_b.upload(queue);
			rDs_{a01b01}.upload(queue);

			dim_1.upload(queue);
			dim_2.upload(queue);

			constexpr bool C0_left = {get_C0_left(formulas[1], formulas[2])};
			GPU_Backend::gemmVbatched2s(
				{Tx_1}, {Ty_1},
				dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_1}, {rDs_y_1},
				GPU_Backend::Type_to_GPU<Tdata>{{0}}, rDs_tmp.d_array,
				{Tx_2}, {Ty_2},
				dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_2},
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, rDs_output.d_array,
				C0_left,
				rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
			GPU_Backend::sync(queue);

			rDs_output.download(Ds_result, queue, fac_add_Ds);
		}} break; // end case {ab_ab}
		"""
		),
		"\t" * 4,
	)
	return content


def content_a01b01_a2b2(ab_ab, formulas):
	ab_ab_tmp = ab_ab.split("_")
	a01b01 = [i for i in ab_ab_tmp if not "2" in i][0]
	a, b = get_ab(formulas)
	Tx_0, Ty_0, dim_m_0_brace, dim_n_0_brace, dim_k_0_brace, dim_m_0_raw, dim_n_0_raw, rDs_x_0, rDs_y_0 = get_gemm_param(formulas[0])
	Tx_1, Ty_1, dim_m_1_brace, dim_n_1_brace, dim_k_1_brace, dim_m_1_raw, dim_n_1_raw, rDs_x_1, rDs_y_1 = get_gemm_param(formulas[1], other="mul")
	Tx_2, Ty_2, dim_m_2_brace, dim_n_2_brace, dim_k_2_brace, dim_m_2_raw, dim_n_2_raw, rDs_x_2, rDs_y_2 = get_gemm_param(formulas[2], other="tmp")

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

			GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_a2b2;
			GPU_Data::Mul<TA, TAC, Tdata> rDs_mul;
			GPU_Data::Tmp<TA, TAC, Tdata> rDs_tmp;
			GPU_Data::Output<TA, TAC, Tdata> rDs_output;

			Dim_mnk dim_0, dim_1, dim_2;

			// {formulas[0]}
			// D_mul = D_a * D_a2b2
			#pragma omp parallel
			{{
				for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
				{{
					const TA &Aa01 = list_Aa01[ia01];
					if(this->filter_atom->filter_for1(label,Aa01))	continue;
					#pragma omp for schedule(dynamic) nowait
					for (std::size_t ia2 = 0; ia2 < list_Aa2.size(); ++ia2)
					{{
						const TAC &Aa2 = list_Aa2[ia2];
						if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
						const Tensor<Tdata> &D_a = {get_D_a(a)};
						if(D_a.empty())	continue;
						for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
						{{
							const TAC &Ab2 = list_Ab2[ib2];
							if(this->filter_atom->filter_for31(label,Aa01,Aa2,Ab2))	continue;
							const Tensor<Tdata> &D_a2b2 = tools.get_Ds_ab(Label::ab::a2b2, Aa2, Ab2);
							if(D_a2b2.empty())	continue;

							#pragma omp critical(rDs_mul)
							{{
								rDs_a.insert(Aa01, Aa2, D_a);
								rDs_a2b2.insert(Aa2.first, {{Ab2.first, (Ab2.second - Aa2.second) % period}}, D_a2b2);
								rDs_mul.insert_1st(Aa01, Ab2, {{{dim_m_0_raw}, {dim_n_0_raw}}});
								dim_0.input({dim_m_0_brace}, {dim_n_0_brace}, {dim_k_0_brace});
							}}
							rDs_a.insert_data(D_a);
							rDs_a2b2.insert_data(D_a2b2);
						}} // end for Ab2
					}} // end for Aa2
				}} // end for Aa01
			}} // end omp parallel

			rDs_a.upload(queue);
			rDs_a2b2.upload(queue);
			rDs_mul.upload_1st(queue);
			dim_0.upload(queue);

			GPU_Backend::gemmVbatched(
				{Tx_0}, {Ty_0},
				dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_0}, {rDs_y_0},
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, rDs_mul.d_array_1,
				rDs_mul.h_array_1.size(), queue);
			GPU_Backend::sync(queue);

			// D_result = D_mul * D_{a01b01} * D_b
			GPU_Data::Input<TA, TAC, Tdata> rDs_b, rDs_{a01b01};

			#pragma omp parallel
			{{
				for (std::size_t ia01 = 0; ia01 < list_Aa01.size(); ++ia01)
				{{
					const TA &Aa01 = list_Aa01[ia01];
					if (this->filter_atom->filter_for1(label, Aa01))	continue;
					#pragma omp for schedule(dynamic) nowait
					for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01)
					{{
						const TAC &Ab01 = list_Ab01[ib01];
						if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
						const Tensor<Tdata> &D_{a01b01} = tools.get_Ds_ab(Label::ab::{a01b01}, Aa01, Ab01);
						if(D_{a01b01}.empty())	continue;
						for (std::size_t ib2 = 0; ib2 < list_Ab2.size(); ++ib2)
						{{
							const TAC &Ab2 = list_Ab2[ib2];
							if (this->filter_atom->filter_for32(label, Aa01, Ab01, Ab2))	continue;
							const Tensor<Tdata> &D_b = {get_D_b(b)};
							if(D_b.empty())	continue;
							const GPU_Data::Pack &D_mul = rDs_mul.find_2nd(Aa01, Ab2);
							if (!D_mul.exist)	continue;

							#pragma omp critical(rDs_output)
							{{
								rDs_b.insert(Ab01.first, {{Ab2.first, (Ab2.second - Ab01.second) % period}}, D_b);
								rDs_{a01b01}.insert(Aa01, Ab01, D_{a01b01});
								rDs_mul.insert_2nd(D_mul);
								const GPU_Data::Pack &D_tmp = rDs_tmp.insert({{{dim_m_1_raw}, {dim_n_1_raw}}});
								rDs_output.insert(Aa01, Ab01, {{{dim_m_2_raw}, {dim_n_2_raw}}});
								dim_1.input({dim_m_1_brace}, {dim_n_1_brace}, {dim_k_1_brace});
								dim_2.input({dim_m_2_brace}, {dim_n_2_brace}, {dim_k_2_brace});
							}}
							rDs_b.insert_data(D_b);
							rDs_{a01b01}.insert_data(D_{a01b01});
						}} // end for Aa01
					}} // end for Ab2
				}} // end for Ab01
			}} // end omp parallel

			rDs_b.upload(queue);
			rDs_{a01b01}.upload(queue);
			rDs_mul.upload_2nd(queue);
			const std::vector<GPU_Backend::Int> rDs_tmp_segments_size = rDs_tmp.upload(memory_limit, queue);
			rDs_output.upload(queue);

			dim_1.upload(queue);
			dim_2.upload(queue);

			constexpr bool C0_left = {get_C0_left(formulas[1], formulas[2])};
			GPU_Backend::gemmVbatched2s(
				{Tx_1}, {Ty_1},
				dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_1}, {rDs_y_1},
				GPU_Backend::Type_to_GPU<Tdata>{{0}}, rDs_tmp.d_array,
				{Tx_2}, {Ty_2},
				dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_2},
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, rDs_output.d_array,
				C0_left,
				rDs_output.h_array.size(), rDs_tmp_segments_size, queue);
			GPU_Backend::sync(queue);

			rDs_output.download(Ds_result, queue, fac_add_Ds);
		}} break; // end case {ab_ab}
		"""), "\t" * 4,
	)
	return content


def content_a01b2_a2b01(ab_ab, formulas):
	ab_ab_tmp = ab_ab.split("_")
	a01b2 = [i for i in ab_ab_tmp if "b2" in i][0]
	a2b01 = [i for i in ab_ab_tmp if "a2" in i][0]
	a, b = get_ab(formulas)
	Tx_0, Ty_0, dim_m_0_brace, dim_n_0_brace, dim_k_0_brace, dim_m_0_raw, dim_n_0_raw, rDs_x_0, rDs_y_0 = get_gemm_param(formulas[0])
	Tx_1, Ty_1, dim_m_1_brace, dim_n_1_brace, dim_k_1_brace, dim_m_1_raw, dim_n_1_raw, rDs_x_1, rDs_y_1 = get_gemm_param(formulas[1])
	Tx_2, Ty_2, dim_m_2_brace, dim_n_2_brace, dim_k_2_brace, dim_m_2_raw, dim_n_2_raw, rDs_x_2, rDs_y_2 = get_gemm_param(formulas[2], other=["mul_2","mul_1"])

	content = textwrap.indent( textwrap.dedent(f"""\
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

			GPU_Data::Input<TA, TAC, Tdata> rDs_a, rDs_b, rDs_{a01b2}, rDs_{a2b01};
			GPU_Data::Mul<TA, TAC, Tdata> rDs_mul_1, rDs_mul_2;
			GPU_Data::Output<TA, TAC, Tdata> rDs_output;

			Dim_mnk dim_0, dim_1, dim_2;

			// D_mul1 = D_b * D_{a01b2}
			// {formulas[0]}
			#pragma omp parallel
			{{
				for (std::size_t ib01 = 0; ib01 < list_Ab01.size(); ++ib01)
				{{
					const TAC &Ab01 = list_Ab01[ib01];
					if(this->filter_atom->filter_for1(label,Ab01))	continue;
					#pragma omp for schedule(dynamic) nowait
					for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
					{{
						const TAC &Ab2 = list_Ab2[ib2];
						if(this->filter_atom->filter_for2(label,Ab01,Ab2))	continue;
						const Tensor<Tdata> &D_b = {get_D_b(b)};
						if(D_b.empty())	continue;
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
							const Tensor<Tdata> &D_{a01b2} = tools.get_Ds_ab(Label::ab::{a01b2}, Aa01, Ab2);
							if(D_{a01b2}.empty())	continue;

							#pragma omp critical(rDs_mul_1)
							{{
								rDs_b.insert(Ab01.first, {{Ab2.first, (Ab2.second - Ab01.second) % period}}, D_b);
								rDs_{a01b2}.insert(Aa01, Ab2, D_{a01b2});
								rDs_mul_1.insert_1st(Aa01, Ab01, {{{dim_m_0_raw}, {dim_n_0_raw}}});
								dim_0.input({dim_m_0_brace}, {dim_n_0_brace}, {dim_k_0_brace});
							}}
							rDs_b.insert_data(D_b);
							rDs_{a01b2}.insert_data(D_{a01b2});
						}} // end for Ab2
					}} // end for Ab01
				}} // end for Aa01
			}} // end omp parallel

			rDs_b.upload(queue);
			rDs_{a01b2}.upload(queue);
			rDs_mul_1.upload_1st(queue);
			dim_0.upload(queue);

			GPU_Backend::gemmVbatched(
				{Tx_0}, {Ty_0},
				dim_0.m.data(), dim_0.n.data(), dim_0.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_0}, {rDs_y_0},
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, rDs_mul_1.d_array_1,
				rDs_mul_1.h_array_1.size(), queue);
			GPU_Backend::sync(queue);

			// D_mul2 = D_{a2b01} * D_a
			// {formulas[1]}
			#pragma omp parallel
			{{
				for (std::size_t ia01 = 0; ia01 < list_Aa01.size(); ++ia01)
				{{
					const TA &Aa01 = list_Aa01[ia01];
					if(this->filter_atom->filter_for1(label,Aa01))	continue;
					#pragma omp for schedule(dynamic) nowait
					for(std::size_t ia2=0; ia2<list_Aa2.size(); ++ia2)
					{{
						const TAC &Aa2 = list_Aa2[ia2];
						const Tensor<Tdata> &D_a = {get_D_a(a)};
						if(D_a.empty())	continue;
						if(this->filter_atom->filter_for2(label,Aa01,Aa2))	continue;
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for32(label, Aa01, Ab01, Aa2))	continue;
							const Tensor<Tdata> &D_{a2b01} = tools.get_Ds_ab(Label::ab::{a2b01}, Aa2, Ab01);
							if(D_{a2b01}.empty())	continue;

							#pragma omp critical(rDs_mul_2)
							{{
								rDs_a.insert(Aa01, Aa2, D_a);
								rDs_{a2b01}.insert(Aa2.first, {{Ab01.first, (Ab01.second - Aa2.second) % period}}, D_{a2b01});
								rDs_mul_2.insert_1st(Aa01, Ab01, {{{dim_m_1_raw}, {dim_n_1_raw}}});
								dim_1.input({dim_m_1_brace}, {dim_n_1_brace}, {dim_k_1_brace});
							}}
							rDs_a.insert_data(D_a);
							rDs_{a2b01}.insert_data(D_{a2b01});
						}} // end for Aa2
					}} // end for Aa01
				}} // end for Ab01
			}} // end omp parallel

			rDs_a.upload(queue);
			rDs_{a2b01}.upload(queue);
			rDs_mul_2.upload_1st(queue);
			dim_1.upload(queue);

			GPU_Backend::gemmVbatched(
				{Tx_1}, {Ty_1},
				dim_1.m.data(), dim_1.n.data(), dim_1.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_1}, {rDs_y_1},
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, rDs_mul_2.d_array_1,
				rDs_mul_2.h_array_1.size(), queue);
			GPU_Backend::sync(queue);

			// D_result = D_mul2 * D_mul1
			// {formulas[2]}
			#pragma omp parallel
			{{
				for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
				{{
					const TA &Aa01 = list_Aa01[ia01];
					if (this->filter_atom->filter_for1(label, Aa01))	continue;
					#pragma omp for schedule(dynamic) nowait
					for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
					{{
						const TAC &Ab01 = list_Ab01[ib01];
						if (this->filter_atom->filter_for2(label, Aa01, Ab01))	continue;
						const GPU_Data::Pack &D_mul_1 = rDs_mul_1.find_2nd(Aa01, Ab01);
						if (!D_mul_1.exist)	continue;
						const GPU_Data::Pack &D_mul_2 = rDs_mul_2.find_2nd(Aa01, Ab01);
						if (!D_mul_2.exist)	continue;

						#pragma omp critical(rDs_output)
						{{
							rDs_mul_1.insert_2nd(D_mul_1);
							rDs_mul_2.insert_2nd(D_mul_2);
							rDs_output.insert(Aa01, Ab01, {{{dim_m_2_raw}, {dim_n_2_raw}}});
							dim_2.input({dim_m_2_brace}, {dim_n_2_brace}, {dim_k_2_brace});
						}}
					}} // end for Ab01
				}} // end for Aa01
			}} // end omp parallel

			rDs_mul_1.upload_2nd(queue);
			rDs_mul_2.upload_2nd(queue);
			rDs_output.upload(queue);
			dim_2.upload(queue);

			GPU_Backend::gemmVbatched(
				{Tx_2}, {Ty_2},
				dim_2.m.data(), dim_2.n.data(), dim_2.k.data(),
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, {rDs_x_2}, {rDs_y_2},
				GPU_Backend::Type_to_GPU<Tdata>{{1}}, rDs_output.d_array,
				rDs_output.h_array.size(), queue);
			GPU_Backend::sync(queue);

			rDs_output.download(Ds_result, queue, fac_add_Ds);
		}} break; // end case {ab_ab}
		"""
		),
		"\t" * 4,
	)
	return content


def content_prefix():
	content = textwrap.dedent(
		"""\
		// =================================
		//  Author: Laiyuan Yang, Peize Lin
		//  date: 2026.01.01
		// =================================

		// This file is automatically generated from tools/cal_loop3/cal_loop3_GPU.py

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
			const Tdata fac_add_Ds)
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

			for(const Label::ab_ab &label : labels)
			{
				const std::vector<TA>  list_Aa01_Da = LRI_Cal_Aux::filter_list_map( this->parallel->list_A[Label_Tools::to_Aab_Aab(label)].a01, data_wrapper(Label::ab::a).Ds_ab );
				const std::vector<TAC> list_Ab01_Db = LRI_Cal_Aux::filter_list_map( this->parallel->list_A[Label_Tools::to_Aab_Aab(label)].b01, data_wrapper(Label::ab::b).Ds_ab );
				const std::vector<TAC> list_Aa2_Da  = LRI_Cal_Aux::filter_list_set( this->parallel->list_A[Label_Tools::to_Aab_Aab(label)].a2,  data_wrapper(Label::ab::a).index_Ds_ab[0] );
				const std::vector<TAC> list_Ab2_Db  = LRI_Cal_Aux::filter_list_set( this->parallel->list_A[Label_Tools::to_Aab_Aab(label)].b2,  data_wrapper(Label::ab::b).index_Ds_ab[0] );
				switch(label)
				{
		"""
	)
	return content


def content_suffix():
	content = textwrap.dedent(
		"""\
					default:
						throw std::invalid_argument(std::string(__FILE__)+std::to_string(__LINE__));
				} // end switch(label)
			} // end for label

			LRI_Cal_Aux::destroy_Ds_result(Ds_result);
			#ifdef __MKL_RI
			mkl_set_num_threads(mkl_threads);
			#endif

		}	// end LRI::cal_loop3()

		}	// end namespace RI

		#endif
		"""
	)
	return content


def print_all():

	a01b01_a01b01 = {
		"a0b0_a1b1": [
			"b1b2a0 = b0b1b2 * a0b0",
			"b2a0a1 = b1b2a0 * a1b1",
			"a2b2 = a0a1a2 * b2a0a1",
		],
		"a0b1_a1b0": [
			"b1b2a1 = b0b1b2 * a1b0",
			"b2a1a0 = b1b2a1 * a0b1",
			"b2a2 = a1a0a2 * b2a1a0",
		],
	}

	a01b01_a01b2 = {
		"a0b0_a1b2": [
			"b0b1a1 = b0b1b2 * a1b2",
			"b1a1a0 = b0b1a1 * a0b0",
			"a2b1 = a1a0a2 * b1a1a0",
		],
		"a0b1_a1b2": [
			"a1b0b1 = a1b2 * b0b1b2",
			"a0a1b0 = a0b1 * a1b0b1",
			"a2b0 = a0a1a2 * a0a1b0",
		],
		"a0b2_a1b0": [
			"b0b1a0 = b0b1b2 * a0b2",
			"b1a0a1 = b0b1a0 * a1b0",
			"a2b1 = a0a1a2 * b1a0a1",
		],
		"a0b2_a1b1": [
			"a0b0b1 = a0b2 * b0b1b2",
			"a1a0b0 = a1b1 * a0b0b1",
			"a2b0 = a1a0a2 * a1a0b0",
		],
	}

	a01b01_a2b01 = {
		"a0b0_a2b1": [
			"b1a1a0 = a2b1 * a1a0a2",
			"b0b1a1 = a0b0 * b1a1a0",
			"a1b2 = b0b1a1 * b0b1b2",
		],
		"a0b1_a2b0": [
			"a0a1b0 = a0a1a2 * a2b0",
			"a1b0b1 = a0a1b0 * a0b1",
			"a1b2 = a1b0b1 * b0b1b2",
		],
		"a1b0_a2b1": [
			"b1a0a1 = a2b1 * a0a1a2",
			"b0b1a0 = a1b0 * b1a0a1",
			"a0b2 = b0b1a0 * b0b1b2",
		],
		"a1b1_a2b0": [
			"a1a0b0 = a1a0a2 * a2b0",
			"a0b0b1 = a1a0b0 * a1b1",
			"a0b2 = a0b0b1 * b0b1b2",
		],
	}

	a01b01_a2b2 = {
		"a0b0_a2b2": [
			"b2a1a0 = a2b2 * a1a0a2",
			"b0b2a1 = a0b0 * b2a1a0",
			"a1b1 = b0b2a1 * b1b0b2",
		],
		"a0b1_a2b2": [
			"b2a1a0 = a2b2 * a1a0a2",
			"b1b2a1 = a0b1 * a2a1a0",
			"a1b0 = b1b2a1 * b0b1b2",
		],
		"a1b0_a2b2": [
			"b2a0a1 = a2b2 * a0a1a2",
			"b0b2a0 = a1b0 * b2a0a1",
			"a0b1 = b0b2a0 * b1b0b2",
		],
		"a1b1_a2b2": [
			"b2a0a1 = a2b2 * a0a1a2",
			"b1b2a0 = a1b1 * b2a0a1",
			"a0b0 = b1b2a0 * b0b1b2",
		],
	}

	a01b2_a2b01 = {
		"a1b2_a2b1": [
			"b0b1a1 = b0b1b2 * a1b2",
			"b1a1a0 = a2b1 * a1a0a2",
			"a0b0 = b1a1a0 * b0b1a1",
		],
		"a0b2_a2b0": [
			"a0b0b1 = a0b2 * b0b1b2",
			"a1a0b0 = a1a0a2 * a2b0",
			"a1b1 = a1a0b0 * a0b0b1",
		],
		"a0b2_a2b1": [
			"b0b1a0 = b0b1b2 * a0b2",
			"b1a0a1 = a2b1 * a0a1a2",
			"a1b0 = b1a0a1 * b0b1a0",
		],
		"a1b2_a2b0": [
			"a1b0b1 = a1b2 * b0b1b2",
			"a0a1b0 = a0a1a2 * a2b0",
			"a0b1 = a0a1b0 * a1b0b1",
		],
	}

	with open("../../include/RI/ri/LRI-cal_loop3_GPU.hpp","w") as file:
		print(content_prefix(), file=file)
		for Aab_Aab in [
			"a01b01_a01b01",
			"a01b01_a01b2",
			"a01b01_a2b01",
			"a01b01_a2b2",
			"a01b2_a2b01",
		]:
			print(f"			  // Aab_Aab::{Aab_Aab}\n", file=file)
			for ab_ab, formulas in eval(Aab_Aab).items():
				print(eval("content_" + Aab_Aab)(ab_ab, formulas), file=file)
		print(content_suffix(), file=file)


print_all()
