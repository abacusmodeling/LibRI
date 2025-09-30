#pragma once
#include "./Exx.h"

namespace RI
{
// Nothing different from Exx, 
// Except the two density matrices can be different
template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
class  LR : public Exx<TA,Tcell,Ndim,Tdata>
{
public:
	LR(const std::string method = "cvc") { this->method = method; }	// Exx default mehtod: "loop3"
	LR(Exx<TA, Tcell, Ndim, Tdata>&& exx) : Exx<TA, Tcell, Ndim, Tdata>::Exx(std::move(exx)) {}
	using Exx<TA,Tcell,Ndim,Tdata>::cal_force;
	// New function: cal_force with two different density matrices
	void cal_force(const std::map<TA, std::map<std::pair<TA, std::array<Tcell, Ndim>>, Tensor<Tdata>>>& Ds_left,
		const std::array<std::string, 5>& save_names_suffix = { "","","","","" })	// "Cs","Vs","Ds","dCs","dVs"
	{
		// The only difference from Exx::cal_force: save Ds_left if not empty
		if (!Ds_left.empty())
			this->post_2D.saves["Ds_" + save_names_suffix[2]] = this->post_2D.set_tensors_map2(Ds_left);
		this->cal_force(save_names_suffix);
	}
};

}