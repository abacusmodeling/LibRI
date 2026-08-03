// ===================
//  Author: Ziqing Guan
//  date: 2026.08.02
// ===================

#pragma once

#include "RI/global/Tensor.h"
#include <mpi.h>
#include <cassert>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <vector>

#define REF_PRECISION 9   // significant digits when writing reference file
#define REF_TOLERANCE 1e-7 // tolerance when comparing against reference

namespace Test_Helpers
{

using Tdata = std::complex<double>;
using TA = int;
using Tcell = int;
constexpr int Ndim = 3;
using TAC = std::pair<TA,std::array<Tcell,Ndim>>;
using Tk = std::array<double,Ndim>;

// ====== type aliases ======
using TensorMap  = std::map<int, std::map<int, RI::Tensor<Tdata>>>;
using HartreeMap = std::map<TA, std::map<TA, std::map<int, RI::Tensor<Tdata>>>>;

// ====== write (recursive dispatch: map -> recurse, Tensor -> leaf) ======

// leaf: Tensor
template<typename T>
inline void write_map(std::ostream &os, const RI::Tensor<T> &t)
{
	os << t.shape.size();
	for (auto d : t.shape) os << " " << d;
	std::size_t total = 1;
	for (auto d : t.shape) total *= d;
	for (std::size_t i = 0; i < total; ++i)
		os << " " << std::setprecision(REF_PRECISION) << std::scientific
		   << t.ptr()[i].real() << " " << t.ptr()[i].imag();
	os << "\n";
}

// recurse: map
template<typename K, typename V>
inline void write_map(std::ostream &os, const std::map<K, V> &m)
{
	os << m.size() << "\n";
	for (auto &[k, v] : m) {
		os << k << " ";
		write_map(os, v);
	}
}

// ====== read ======

// leaf: Tensor
template<typename T>
inline void read_map(std::istream &is, RI::Tensor<T> &t)
{
	std::size_t ndim; is >> ndim;
	std::vector<std::size_t> shape(ndim);
	std::size_t total = 1;
	for (std::size_t d = 0; d < ndim; ++d) {
		is >> shape[d]; total *= shape[d];
	}
	t = RI::Tensor<T>(shape);
	for (std::size_t i = 0; i < total; ++i) {
		double re, im; is >> re >> im; t.ptr()[i] = T(re, im);
	}
}

// recurse: map
template<typename K, typename V>
inline void read_map(std::istream &is, std::map<K, V> &m)
{
	m.clear();
	std::size_t n; is >> n;
	for (std::size_t i = 0; i < n; ++i) {
		K key; is >> key;
		read_map(is, m[key]);
	}
}

// ====== compare ======

// leaf: Tensor
template<typename T>
inline void compare_map(const RI::Tensor<T> &a, const RI::Tensor<T> &b)
{
	assert(a.shape.size() == b.shape.size());
	std::size_t total = 1;
	for (std::size_t d = 0; d < a.shape.size(); ++d) {
		assert(a.shape[d] == b.shape[d]);
		total *= a.shape[d];
	}
	for (std::size_t i = 0; i < total; ++i)
		assert(std::abs(a.ptr()[i] - b.ptr()[i]) < REF_TOLERANCE);
}

// recurse: map
template<typename K, typename V>
inline void compare_map(const std::map<K, V> &a, const std::map<K, V> &b)
{
	assert(a.size() == b.size());
	for (auto &[k, va] : a) {
		assert(b.count(k));
		compare_map(va, b.at(k));
	}
}

// ====== merge (for MPI gather) ======

// base: non-map value — move individual entries
template<typename K, typename V>
inline void merge_into(std::map<K, V> &target, std::map<K, V> &&source)
{
	for (auto &[k, v] : source)
		target[k] = std::move(v);
}

// recurse: value is another map — merge inner maps element by element
template<typename K, typename K2, typename V2>
inline void merge_into(std::map<K, std::map<K2, V2>> &target,
                       std::map<K, std::map<K2, V2>> &&source)
{
	for (auto &[k, inner] : source)
		merge_into(target[k], std::move(inner));
}

// ====== MPI gather ======

template<typename M>
inline M gather_map(const M &local, int root, MPI_Comm comm, int tag = 0)
{
	int rank, nproc;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	std::ostringstream oss;
	write_map(oss, local);
	std::string data = oss.str();
	int len = static_cast<int>(data.size());

	if (rank == root)
	{
		M combined = local;
		for (int r = 0; r < nproc; ++r)
		{
			if (r == root) continue;
			int remote_len;
			MPI_Recv(&remote_len, 1, MPI_INT, r, tag, comm, MPI_STATUS_IGNORE);
			std::string remote_data(remote_len, '\0');
			MPI_Recv(&remote_data[0], remote_len, MPI_CHAR, r, tag, comm, MPI_STATUS_IGNORE);
			std::istringstream iss(remote_data);
			M remote;
			read_map(iss, remote);
			merge_into(combined, std::move(remote));
		}
		return combined;
	}
	else
	{
		MPI_Send(&len, 1, MPI_INT, root, tag, comm);
		MPI_Send(data.data(), len, MPI_CHAR, root, tag, comm);
		return {};
	}
}

}
