#pragma once

#include "Exx_Post_2D.h"

#include "../global/Global_Func-1.h"
#include "../global/Map_Operator-2.h"
#include "../global/Map_Operator-3.h"
#include "../global/MPI_Wrapper.h"
#include "../comm/mix/Communicate_Tensors_Map_Judge.h"
#include "../distribute/Distribute_Equally.h"

#include <vector>


template<typename TA, typename TC, typename Tdata> template<typename Tatom_pos>
void Exx_Post_2D<TA,TC,Tdata>::set_parallel(
	const MPI_Comm &mpi_comm_in,
	const std::map<TA,Tatom_pos> &atoms_pos,
	const TC &period)
{
	this->mpi_comm = mpi_comm_in;

	constexpr size_t num_index = 2;
	const std::vector<TA> atoms_vec = Global_Func::map_key_to_vec(atoms_pos);

	const std::pair<std::vector<TA>, std::vector<std::vector<TAC>>>
		atoms_split_list = Distribute_Equally::distribute_atoms_periods(
			this->mpi_comm, atoms_vec, period, num_index);

	this->list_Aa = Global_Func::to_set(atoms_split_list.first);
	this->list_Ab = Global_Func::to_set(atoms_split_list.second[0]);
}


template<typename TA, typename TC, typename Tdata>
auto Exx_Post_2D<TA,TC,Tdata>::set_tensors_map2( const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds_in ) const
-> std::map<TA,std::map<TAC,Tensor<Tdata>>>
{
	return Communicate_Tensors_Map_Judge::comm_map2(this->mpi_comm, Ds_in, this->list_Aa, this->list_Ab);
}


template<typename TA, typename TC, typename Tdata>
Tdata Exx_Post_2D<TA,TC,Tdata>::cal_energy(
	const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds,
	const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Hs) const
{
	typedef Tdata(*Tfunc_dotc)(const Tensor<Tdata>&, const Tensor<Tdata>&);
	const std::function<Tdata(const Tensor<Tdata>&, const Tensor<Tdata>&)>
		dotc = static_cast<Tfunc_dotc>(Blas_Interface::dotc);
	const std::map<TA, std::map<TAC, Tdata>>
		E_map = Map_Operator::zip_intersection(Ds, Hs, dotc);

	const std::function<Tdata(const Tdata&, const Tdata&)>
		plus = std::plus<Tdata>();
	Tdata E = Map_Operator::reduce(E_map, Tdata(0), plus);

	MPI_Wrapper::mpi_allreduce( E, MPI_SUM, this->mpi_comm );
	return E;
}