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

namespace
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

template <typename T>
void check(const std::string& scalar, T imaginary, int selected, const std::string& backend)
{
    RI::LRI<int, int, 1, T> lri;
    const std::vector<Label> all_labels(RI::Label::array_ab_ab.begin(), RI::Label::array_ab_ab.end());
    const std::map<int, std::array<double, 1>> atoms{{0, {0.0}}, {1, {0.4}}};
    const std::array<std::array<double, 1>, 1> lattice{{{{1.0}}}};
    const Cell period{{3}};
    lri.set_parallel(MPI_COMM_WORLD, atoms, lattice, period, all_labels);
    int label_index = 0;
    for (const auto label : RI::Label::array_ab)
    {
        const bool coefficient = label == RI::Label::ab::a || label == RI::Label::ab::b;
        const RI::Shape_Vector shape = coefficient ? RI::Shape_Vector{2, 2, 2} : RI::Shape_Vector{2, 2};
        Tensors<T> tensors;
        for (int left = 0; left < 2; ++left)
            for (int right = 0; right < 2; ++right)
                for (int cell = 0; cell < period[0]; ++cell)
                    tensors[left][{right, {cell}}] = make_tensor(shape, left, right, cell, label_index, imaginary);
        lri.set_tensors_map2(tensors, {label});
        ++label_index;
    }

    // Empty means no reduction in the existing LibRI API. The other sectors
    // exercise unequal atom identities, relative translations, and empty output.
    const std::vector<Sector> sectors{
        {},
        {{{1, 0}, {Cell{{1}}}}},
        {{{0, 1}, {Cell{{0}}, Cell{{-1}}}}, {{1, 1}, {Cell{{1}}}}},
        {{{7, 9}, {Cell{{1}}}}}
    };
    std::vector<RI::LRI_Cal_Mode> modes;
    if (backend != "gpu")
    {
        modes.push_back(RI::LRI_Cal_Mode::CPU);
        modes.push_back(RI::LRI_Cal_Mode::CPU_fine_grained_lock);
    }
#ifdef __GPU_RI
    if (backend != "cpu")
        modes.push_back(RI::LRI_Cal_Mode::GPU);
#endif
    if (modes.empty())
        throw std::runtime_error("Requested backend is not compiled in");

    const int count = static_cast<int>(all_labels.size());
    for (int index = 0; index <= count; ++index)
    {
        if (selected >= 0 && index != selected)
            continue;
        std::vector<Label> labels;
        std::string name;
        if (index == count)
        {
            labels = all_labels;
            name = "all";
        }
        else
        {
            labels.push_back(all_labels[index]);
            name = RI::Label_Tools::get_name(all_labels[index]);
        }
        lri.filter_atom = std::make_shared<RI::Filter_Atom<int, AtomCell>>();
        lri.cal_mode = RI::LRI_Cal_Mode::CPU;
        Tensors<T> full;
        lri.cal_loop3(labels, full, T(1));
        if (full.size() != 2)
            throw std::runtime_error("CPU reference must contain both atoms");
        for (const auto& left : full)
            if (left.second.size() != 6)
                throw std::runtime_error("CPU reference must contain all atom/cell blocks");

        for (std::size_t sector_index = 0; sector_index < sectors.size(); ++sector_index)
        {
            const auto& sector = sectors[sector_index];
            Tensors<T> expected;
            for (const auto& left : full)
                for (const auto& right : left.second)
                {
                    const auto pair = sector.find({left.first, right.first.first});
                    const bool keep = sector.empty()
                                      || (pair != sector.end() && pair->second.count(right.first.second));
                    if (keep)
                        expected[left.first][right.first] = right.second.copy();
                }
            lri.filter_atom = std::make_shared<RI::Filter_Atom_Symmetry<int, Cell, T>>(period, sector);
            for (const auto mode : modes)
            {
                lri.cal_mode = mode;
                const std::string context = scalar + " label=" + name + " sector="
                                            + std::to_string(sector_index) + " mode="
                                            + std::to_string(static_cast<int>(mode));
                std::cout << "CHECK " << context << std::endl;
                Tensors<T> actual;
                lri.cal_loop3(labels, actual, T(1));
                compare(actual, expected, context);

                // Seed every block, including those outside the selected sector.
                // Filtering must preserve them while signed/zero increments work.
                Tensors<T> seed;
                for (const auto& left : full)
                    for (const auto& right : left.second)
                        seed[left.first][right.first] = make_tensor(right.second.shape, left.first,
                            right.first.first, right.first.second[0], 0, imaginary);
                actual = copy_tensors(seed);
                double sum = 0;
                for (const double factor : {-0.5, 0.0, 1.25})
                {
                    lri.cal_loop3(labels, actual, T(factor));
                    sum += factor;
                    Tensors<T> accumulated = copy_tensors(seed);
                    for (const auto& left : expected)
                        for (const auto& right : left.second)
                        {
                            auto& target = accumulated.at(left.first).at(right.first);
                            for (std::size_t i = 0; i < target.data->size(); ++i)
                                (*target.data)[i] += T(sum) * (*right.second.data)[i];
                        }
                    compare(actual, accumulated, context + " factor=" + std::to_string(factor));
                }
            }
        }
        std::cout << "PASS " << scalar << " label=" << name << std::endl;
    }
}
} // namespace

int main(int argc, char** argv)
{
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS)
        return 1;
    int status = 0;
    try
    {
        if (RI::MPI_Wrapper::mpi_get_size(MPI_COMM_WORLD) != 1 || provided < MPI_THREAD_MULTIPLE)
            throw std::runtime_error("This test requires one MPI rank and MPI_THREAD_MULTIPLE");
        const int selected = argc > 1 ? std::stoi(argv[1]) : -1;
        const std::string backend = argc > 2 ? argv[2] : "all";
        const std::string scalar = argc > 3 ? argv[3] : "all";
        if (selected < -1 || selected > 18 || (backend != "cpu" && backend != "gpu" && backend != "all")
            || (scalar != "real" && scalar != "complex" && scalar != "all"))
            throw std::runtime_error("Usage: test_lri_gpu_symmetry [-1..18] [cpu|gpu|all] [real|complex|all]");
        if (scalar != "complex")
            check<double>("real", 0.0, selected, backend);
        if (scalar != "real")
            check<std::complex<double>>("complex", {0.0, 1.0}, selected, backend);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    MPI_Finalize();
    return status;
}
