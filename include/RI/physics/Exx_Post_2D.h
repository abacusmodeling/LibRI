#pragma once

#include "../global/Tensor.h"
#include "../global/Global_Func-2.h"

#include <map>
#include <set>
#include <string>
#include <mpi.h>

template<typename TA, typename TC, typename Tdata>
class Exx_Post_2D
{
public:
	using TAC = std::pair<TA,TC>;

	std::set<TA>  list_Aa;
	std::set<TAC> list_Ab;

	std::map<std::string, std::map<TA,std::map<TAC,Tensor<Tdata>>>> saves;
	Tdata energy = 0;

	template<typename Tatom_pos>
	void set_parallel(
		const MPI_Comm &mpi_comm_in,
		const std::map<TA,Tatom_pos> &atoms_pos,
		const TC &period);

	std::map<TA,std::map<TAC,Tensor<Tdata>>>
	set_tensors_map2( const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds_in ) const;

	Tdata cal_energy(
		const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds,
		const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Hs ) const;

public:	// private:
	MPI_Comm mpi_comm;
};

#include "Exx_Post_2D.hpp"