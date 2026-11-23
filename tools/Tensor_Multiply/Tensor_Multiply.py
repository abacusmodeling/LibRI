# ===================
#  Author: Peize Lin
#  date: 2023.08.17
# ===================

import textwrap
import numpy as np

class judge_fix_Error(Exception):
	def __init__(self, error):
		Exception.__init__(self, error)

def generate_cal(Txy, Tx, Ty, file):
	def judge_fix(T, sub):
		size = len(sub)
		if T[:size] == sub:
			return "prefix",T[size:]
		elif T[-size:] == sub:
			return "suffix", T[:-size]
		else:
			raise judge_fix_Error(f"T={T}, sub={sub}")

	contraction = sorted(list(set(Tx + Ty) - set(Txy)))
	fix_x, remain_x = judge_fix(Tx, contraction)
	fix_y, remain_y = judge_fix(Ty, contraction)
	fix_xy, remain_xy = judge_fix(Txy, remain_x)
	assert(fix_xy=="prefix")
	assert(remain_xy == remain_y)

	def get_index(T, fix, remain):
		if fix=="prefix":
			return np.arange(0,len(T),1)[-len(remain):]
		elif fix=="suffix":
			return np.arange(0,len(T),1)[:len(remain)]
	xr = get_index(Tx, fix_x, remain_x)
	yr = get_index(Ty, fix_y, remain_y)
	if fix_x=="prefix":
		con_x = np.arange(0,len(Tx),1)[:len(contraction)]
	elif fix_x=="suffix":
		con_x = np.arange(0,len(Tx),1)[-len(contraction):]

	indentation = 1
	content = textwrap.indent(textwrap.dedent(f"""\
		// Txy({",".join(Txy)}) = Tx({",".join(Tx)}) * Ty({",".join(Ty)})
		template<typename Tdata>
		Tensor<Tdata> {"".join(Txy)}_{"".join(Tx)}_{"".join(Ty)}(const Tensor<Tdata> &Tx, const Tensor<Tdata> &Ty)
		{{
			assert(Tx.shape.size()=={len(Tx)});
			assert(Ty.shape.size()=={len(Ty)});
		"""),
		"\t"*indentation)
	content += textwrap.indent(textwrap.dedent(f"""\
			Tensor<Tdata> Txy({{{", ".join([f"T{s[0]}.shape[{s[1]}]" for s in Txy])}}});
			Blas_Interface::gemm(
				{"'T'" if fix_x=="prefix" else "'N'"}, {"'N'" if fix_y=="prefix" else "'T'"},
				{" * ".join([f"Tx.shape[{i}]" for i in xr])},
				{" * ".join([f"Ty.shape[{i}]" for i in yr])},
				{" * ".join([f"Tx.shape[{i}]" for i in con_x])},
				Tdata(1.0), Tx.ptr(), Ty.ptr(),
				Tdata(0.0), Txy.ptr());
			return Txy;
		}}
		"""),
		"\t"*indentation)

	file.write(content)