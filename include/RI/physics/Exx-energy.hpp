class Exx-energy
{
public:
	std::set<TA>  list_Aa;
	std::set<TAC> list_Ab;
};


template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void Parallel_LRI_Equally<TA,Tcell,Ndim,Tdata>::set_parallel(
	const MPI_Comm &mpi_comm_in,
	const std::map<TA,TatomR> &atomsR,
	const std::array<TatomR,Ndim> &latvec,
	const std::array<Tcell,Ndim> &period_in)
{
	this->mpi_comm = mpi_comm_in;
	this->period = period_in;

	constexpr size_t num_index = 2;
	const std::vector<TA> atoms_vec = Global_Func::map_key_to_vec(atomsR);

	std::pair<std::vector<TA>, std::vector<std::vector<TAC>>>
		atoms_split_list = Distribute_Equally::distribute_atoms_periods(
			mpi_comm, atoms_vec, period, num_index)		
			
	this->list_Aa = Global_Func::to_set(atoms_split_list.first);
	this->list_Ab = Global_Func::to_set(atoms_split_list.second[0]);
}

set_Ds( const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds)
{
	return Communicate_Tensors_Map_Judge::comm_map2(this->mpi_comm, Ds, this->list_Aa, this->list_Ab);
}



cal_E()
{
	Tdata_real E = 0;
	for(const auto &Hs_tmpA : this->Hs)
	{
		const Ds_ptrA = Ds Hs_tmpA.first
		if(Ds_ptrA==Ds.end()) continue;
		for(const auto &Hs_tmpB : Hs_tmpA.second)
		{
			const Ds_ptrB = Ds_ptrA.second Hs_tmpB.first
			if(Ds_ptrB==Ds_ptrA.second.end())	continue;
			E += Hs_tmpB.second dot Ds_ptrB.second.conj()
		}
	}
	MPI_reduce(E);
	return E;
}