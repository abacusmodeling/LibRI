// ===================
//  Author: maki49
//  date: 2026.07.10
// ===================

#pragma once

#include "Parallel_LRI_Equally.h"

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
class Parallel_LRI_Equally_Weighted: public Parallel_LRI_Equally<TA,Tcell,Ndim,Tdata>
{
  public:
	using TC = std::array<Tcell,Ndim>;
	using TAC = std::pair<TA,TC>;
	using Tatom_pos = std::array<double,Ndim>;		// tmp

	Parallel_LRI_Equally_Weighted(
		const std::map<TA,std::size_t> &atoms_weight_in)
		: atoms_weight(atoms_weight_in){}
	// atoms_weight[iA] = the number of atomic orbitals of atom iA, used to balance the load
	//                    across processes. Empty => balance the atom count instead.

  public:	// private:
	const std::map<TA,std::size_t> atoms_weight;

  public:	// private:
	virtual void set_parallel_loop4(
		const std::vector<TA> &atoms_vec);
	virtual void set_parallel_loop3(
		const std::vector<TA> &atoms_vec,
		const std::set<Label::Aab_Aab> &labels);
};

}

#include "Parallel_LRI_Equally_Weighted.hpp"
