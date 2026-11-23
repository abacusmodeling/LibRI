#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <RI/physics/symmetry/Filter_Atom_Symmetry.h>
#include <RI/ri/LRI.h>

namespace tensor_test
{
using Cell = std::array<int, 1>;
using AtomCell = std::pair<int, Cell>;
using Sector = std::map<std::pair<int, int>, std::set<Cell>>;
using Label = RI::Label::ab_ab;

template <typename T>
using Tensors = std::map<int, std::map<AtomCell, RI::Tensor<T>>>;

template <typename T>
RI::Tensor<T> make_tensor(const RI::Shape_Vector& shape, int left, int right,
                          int cell, int label, T imaginary)
{
    RI::Tensor<T> tensor(shape);
    for (std::size_t i = 0; i < tensor.data->size(); ++i)
    {
        // Different atoms and cells must not accidentally produce equal sums.
        const double real = 0.2 + 0.07 * left + 0.03 * right + 0.02 * cell
                            + 0.005 * label + 0.013 * i;
        const double imag = 0.04 + 0.02 * left - 0.01 * right + 0.003 * i;
        (*tensor.data)[i] = T(real) + imaginary * T(imag);
    }
    return tensor;
}

template <typename T>
Tensors<T> copy_tensors(const Tensors<T>& source)
{
    Tensors<T> result;
    for (const auto& left : source)
        for (const auto& right : left.second)
            result[left.first][right.first] = right.second.copy();
    return result;
}

template <typename T>
void compare(const Tensors<T>& actual, const Tensors<T>& expected,
             const std::string& context)
{
    if (actual.size() != expected.size())
        throw std::runtime_error(context + ": output atom count differs");
    for (const auto& left : expected)
    {
        const auto found_left = actual.find(left.first);
        if (found_left == actual.end() || found_left->second.size() != left.second.size())
            throw std::runtime_error(context + ": output block count differs");
        for (const auto& right : left.second)
        {
            const auto found = found_left->second.find(right.first);
            if (found == found_left->second.end())
                throw std::runtime_error(context + ": missing output block");
            const auto& value = found->second;
            const auto& reference = right.second;
            if (value.shape.size() != reference.shape.size()
                || !std::equal(value.shape.begin(), value.shape.end(), reference.shape.begin()))
                throw std::runtime_error(context + ": output shape differs");
            for (std::size_t i = 0; i < reference.data->size(); ++i)
            {
                const double error = std::abs((*value.data)[i] - (*reference.data)[i]);
                if (!std::isfinite(error) || error > 1e-10 * (1.0 + std::abs((*reference.data)[i])))
                {
                    std::cerr << context << " atoms=" << left.first << ',' << right.first.first
                              << " cell=" << right.first.second[0] << " element=" << i
                              << " error=" << error << '\n';
                    throw std::runtime_error(context + ": contraction differs from masked CPU reference");
                }
            }
        }
    }
}

} // namespace tensor_test
