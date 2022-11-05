// ===================
//  Author: Peize Lin
//  date: 2022.07.23
// ===================

#pragma once

#include "Parallel_LRI.h"

namespace RI
{

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
class Parallel_LRI_Equally: public Parallel_LRI<TA,Tcell,Ndim,Tdata>
{
public:
	using TC = std::array<Tcell,Ndim>;
	using TAC = std::pair<TA,TC>;
	using TatomR = std::array<double,Ndim>;		// tmp

	void set_parallel(
		const MPI_Comm &mpi_comm_in,
		const std::map<TA,TatomR> &atomsR,
		const std::array<TatomR,Ndim> &latvec,
		const std::array<Tcell,Ndim> &period_in) override;

	std::map<TA,std::map<TAC,Tensor<Tdata>>> comm_tensors_map2(
		const Label::ab &label,
		const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds) const override;

	const std::vector<TA >& get_list_Aa01()                                                const override { return this->list_Aa01; }
	const std::vector<TAC>& get_list_Aa2 (const TA &Aa01)                                  const override { return this->list_Aa2;  }
	const std::vector<TAC>& get_list_Ab01(const TA &Aa01, const TAC &Aa2)                  const override { return this->list_Ab01; }
	const std::vector<TAC>& get_list_Ab2 (const TA &Aa01, const TAC &Aa2, const TAC &Ab01) const override { return this->list_Ab2;  }

// private:
public:
	std::vector<TA>  list_Aa01;
	std::vector<TAC> list_Aa2;
	std::vector<TAC> list_Ab01;
	std::vector<TAC> list_Ab2;

	MPI_Comm mpi_comm;
	std::array<Tcell,Ndim> period;
};

}

#include "Parallel_LRI_Equally.hpp"