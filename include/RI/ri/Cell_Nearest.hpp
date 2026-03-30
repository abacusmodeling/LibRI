// ===================
//  Author: Peize Lin
//  date: 2022.12.30
// ===================

#pragma once

#include "Cell_Nearest.h"

#include "../global/Array_Operator.h"
#include "../global/Tensor_Algorithm.h"

#include <cmath>
#include <limits>

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tpos, std::size_t Npos>
void Cell_Nearest<TA,Tcell,Ndim,Tpos,Npos>::init(
	const std::map<TA,Tatom_pos> &atoms_pos_in,
	const std::array<Tatom_pos,Ndim> &latvec_in,
	const std::array<Tcell,Ndim> &period_in)
{
	using namespace Array_Operator;
	this->period = period_in;
	for(const auto &atom_pos : atoms_pos_in)
		this->atoms_pos[atom_pos.first] = to_Tensor(atom_pos.second);
	this->latvec = to_Tensor(latvec_in);

	const Tensor<Tpos> least_square_tmp				// shape:{Ndim,Npos}
		= - Tensor_Algorithm::inverse_matrix_heev(this->latvec * this->latvec.transpose()) * this->latvec;

	for(const auto &atoms_pos_x : this->atoms_pos)
	{
		const TA &Ax = atoms_pos_x.first;
		for(const auto &atoms_pos_y : this->atoms_pos)
		{
			const TA &Ay = atoms_pos_y.first;
			const Tensor<Tpos> delta_pos = atoms_pos_y.second - atoms_pos_x.second;
			this->cells_nearest_continuous[Ax][Ay] = to_array<Tpos,Ndim>(least_square_tmp * delta_pos);
		}
	}
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tpos, std::size_t Npos>
auto Cell_Nearest<TA,Tcell,Ndim,Tpos,Npos>::get_cell_nearest_discrete(
	const TA &Ax, const TA &Ay, const std::array<Tcell,Ndim> &cell) const
-> std::array<Tcell,Ndim>
{
	using namespace Array_Operator;
	const std::array<Tpos,Ndim> &cell_nearest_continuous  = this->cells_nearest_continuous.at(Ax).at(Ay);
	const Tensor<Tpos> pos_delta = this->atoms_pos.at(Ay) - this->atoms_pos.at(Ax) + to_Tensor<Tpos>(cell) * this->latvec;

	std::array<Tpos,Ndim> cell_delta_direct;
	for(std::size_t idim=0; idim<Ndim; ++idim)
		cell_delta_direct[idim] = (cell_nearest_continuous[idim] - cell[idim]) / this->period[idim];

	std::array<Tcell,Ndim> cell_nearest;
	Tpos dist_nearest = std::numeric_limits<Tpos>::max();

	constexpr std::size_t mask_total = 1 << Ndim;
	for(std::size_t mask=0; mask<mask_total; ++mask)
	{
		std::array<Tcell,Ndim> cell_candidate;
		for(std::size_t idim=0; idim<Ndim; ++idim)
			cell_candidate[idim] =
				(((mask>>idim)&1) ? std::ceil(cell_delta_direct[idim]) : std::floor(cell_delta_direct[idim]))
				* this->period[idim];

		const Tensor<Tpos> pos_candidate = pos_delta + to_Tensor<Tpos>(cell_candidate) * this->latvec;
		const Tpos dist = pos_candidate.norm(2);
		if(dist < dist_nearest)
		{
			dist_nearest = dist;
			cell_nearest = cell_candidate;
		}
	}
	return cell_nearest + cell;
}

}
