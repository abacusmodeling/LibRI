#pragma once

#include "RI/ri/LRI_k.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>

namespace LRI_K_Atom_Lists_Test
{
using Scalar = std::complex<double>;
using LRI = RI::LRI_k<int, int, 3, Scalar>;
using Tensors = std::map<int, std::map<LRI::TAC, RI::Tensor<Scalar>>>;

inline RI::Tensor<Scalar> tensor(const RI::Shape_Vector &shape, double value)
{
    RI::Tensor<Scalar> result(shape);
    for (auto &element : *result.data)
        element = value;
    return result;
}

inline void setup(LRI &lri)
{
    lri.set_parallel(MPI_COMM_SELF, {{0, {0, 0, 0}}, {1, {0.3, 0, 0}}, {2, {0.7, 0, 0}}},
                     {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {1, 1, 1},
                     RI::Global_Func::to_vector(RI::Label::array_ab_ab));
    Tensors cs, vs;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        {
            cs[a][{b, {0, 0, 0}}] = tensor({1, 1, 1}, 1 + a + 2*b);
            vs[a][{b, {0, 0, 0}}] = tensor({1, 1}, 2 + a + 3*b);
        }
    lri.set_tensors_map2(cs, {RI::Label::ab::a}, {}, "Cs");
    lri.set_tensors_map2(vs, {RI::Label::ab::a0b0}, {}, "Vs");
}

inline void check(const RI::Tensor<Scalar> &actual, double expected)
{
    if (actual.empty() || actual.data->size() != 1)
        throw std::runtime_error("Missing atom-list contraction output");
    const double error = std::abs((*actual.data)[0] - expected);
    if (!std::isfinite(error) || error > 1e-11 * (1 + std::abs(expected)))
        throw std::runtime_error("Atom-list contraction differs from scalar reference");
}

inline void test_hartree()
{
    LRI lri;
    setup(lri);
    const std::vector<int> left = {0, 1, 2}, right = {0, 1};
    std::map<int, std::map<int, std::map<int, RI::Tensor<Scalar>>>> density;
    double m[3] = {}, n[3] = {};
    for (int v : left)
        for (int u : right)
        {
            const double d = 0.1 * (1 + 2*v + u);
            density[v][u][0] = tensor({1, 1}, d);
            m[u] += (1 + u + 2*v) * d;
            m[v] += (1 + v + 2*u) * d;
        }
    for (int mu : left)
        for (int nu : left)
            n[mu] += (2 + mu + 3*nu) * m[nu];

    const auto result = lri.cal_cvcd_k_hartree(
        density, {{0, 0, 0}}, {0}, left, right, left, "Cs", "Vs");
    for (int s : left)
        for (int t : right)
            check(result.at(s).at(t).at(0),
                  (1 + s + 2*t) * n[s] + (1 + t + 2*s) * n[t]);
    std::cout << "PASS Hartree with unequal left/right atom lists" << std::endl;
}

inline void test_cvc()
{
    LRI lri;
    setup(lri);
    std::map<int, std::map<int, RI::Tensor<Scalar>>> cs_mo, psi;
    for (int atom = 0; atom < 3; ++atom)
    {
        cs_mo[0][atom] = tensor({1, 1, 2}, atom + 1);
        psi[0][atom] = tensor({2, 1}, 1);
    }
    const std::vector<std::pair<std::vector<int>, std::vector<int>>> partitions = {
        {{0}, {1}}, {{0}, {1, 2}}, {{0, 1}, {2}}};
    for (const auto &partition : partitions)
        for (bool is_A : {false, true})
        {
            double expected = 0;
            for (int mu : partition.first)
                for (int nu : partition.second)
                    // Each MO coefficient contains its two conjugate contributions.
                    expected += 4 * (mu + 1) * (2 + mu + 3*nu) * (nu + 1);
            const auto result = lri.cal_cvc_mo_k_hartree_onthefly(
                cs_mo, psi, {0}, {0}, partition.first, partition.second,
                {"O", "V", "O", "V"}, 1, 1, "Vs", is_A);
            check(result.at(0).at(0), expected);
        }
    std::cout << "PASS CVC with distinct and unequal left/right atom lists" << std::endl;
}
}
