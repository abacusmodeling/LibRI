# ===================
#  Author: Peize Lin
#  date: 2023.08.17
# ===================

import textwrap

def generate_test(Txy, Tx, Ty, file):
	indentation = 1
	contraction = sorted(list(set(Tx + Ty) - set(Txy)))
	content = textwrap.indent(textwrap.dedent(f"""\
		// Txy({",".join(Txy)}) = Tx({",".join(Tx)}) * Ty({",".join(Ty)})
		template<typename Tdata>
		void {"".join(Txy)}_{"".join(Tx)}_{"".join(Ty)}_test()
		{{
			std::cout<<"Txy({','.join(Txy)}) = Tx({','.join(Tx)}) * Ty({','.join(Ty)})"<<std::endl;
			const std::size_t """),
		"\t"*indentation)
	content += ", ".join([f"{s.upper()}={i+2}" for i,s in enumerate(Txy + contraction)]) + ";\n"
	content += textwrap.indent(textwrap.dedent(f"""\
			const RI::Tensor<Tdata> Tx = init_tensor<Tdata>({{{",".join([s.upper() for s in Tx])}}});
			const RI::Tensor<Tdata> Ty = init_tensor<Tdata>({{{",".join([s.upper() for s in Ty])}}});
			const RI::Tensor<Tdata> Txy = RI::Tensor_Multiply::{"".join(Txy)}_{"".join(Tx)}_{"".join(Ty)}(Tx,Ty);
		"""),
		"\t"*(indentation+1))
	for i,index in enumerate(Txy):
		content += "\t"*(indentation+1+i) + f"FOR_0({index},{index.upper()})\n"
	content += textwrap.indent(textwrap.dedent(f"""\
		{{
			Tdata result = 0;
		"""),
		"\t"*(indentation+1+i))
	for j,index in enumerate(contraction):
		content += "\t"*(indentation+2+i+j) + f"FOR_0({index},{index.upper()})\n"
	content += "\t"*(indentation+3+i+j) + f'result += Tx({",".join(Tx)}) * Ty({",".join(Ty)});\n'
	content += "\t"*(indentation+2+i) + f'assert(Txy({",".join(Txy)}) == result);\n'
	content += "\t"*(indentation+1+i) + "}\n"
	content += "\t"*(indentation) + "}\n"

	file.write(content)
