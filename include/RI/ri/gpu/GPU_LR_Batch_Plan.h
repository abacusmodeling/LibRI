#ifndef RI_GPU_LR_BATCH_PLAN_H
#define RI_GPU_LR_BATCH_PLAN_H

#ifdef __GPU_RI

#include "GPU_Data_Input.h"
#include "GPU_Data_Mul.h"
#include "GPU_Data_Tmp.h"
#include "GPU_Data_Output.h"
#include "Dim.h"
#include "../Data_Pack.h"
#include "../../parallel/Parallel_LRI.h"
#include <algorithm>
#include <initializer_list>
#include <stdexcept>

namespace RI
{

// Symbolic layout of the two GEMM stages for one LR contraction label.
// Offsets and dimensions persist; each call binds them to current C, V, D values.
template<typename TA, typename TAC, typename Tdata>
struct Gpu_LR_Case_Plan
{
    using Input = GPU_Data::Input<TA, TAC, Tdata>;
    using Mul = GPU_Data::Mul<TA, TAC, Tdata>;
    using Tmp = GPU_Data::Tmp<TA, TAC, Tdata>;
    using Output = GPU_Data::Output<TA, TAC, Tdata>;
    using Matrix_Map = std::map<TA, std::map<TAC, Tensor<Tdata>>>;
    using Pack_Map = std::map<TA, std::map<TAC, GPU_Data::Pack>>;

    struct Stage
    {
        bool ready = false;
        std::vector<typename Input::Layout> inputs;
        Pack_Map mul_map;
        std::vector<GPU_Data::Pack> mul_first;
        std::vector<GPU_Data::Pack> mul_second;
        std::size_t mul_size = 0;
        std::vector<GPU_Data::Pack> tmp_array;
        std::size_t tmp_size = 0;
        Pack_Map output_map;
        std::vector<GPU_Data::Pack> output_array;
        std::size_t output_size = 0;
        std::vector<int> m0;
        std::vector<int> n0;
        std::vector<int> k0;
        std::vector<int> m1;
        std::vector<int> n1;
        std::vector<int> k1;
    } first, second;

    static void save_inputs(Stage &stage, std::initializer_list<Input*> inputs)
    {
        stage.inputs.clear();
        for(Input *input : inputs)
            stage.inputs.push_back(input->save_layout());
    }

    static void load_inputs(const Stage &stage,
                            std::initializer_list<Input*> inputs,
                            std::initializer_list<const Matrix_Map*> sources)
    {
        if(inputs.size() != stage.inputs.size() || sources.size() != inputs.size())
            throw std::logic_error("GPU LR batch plan input count changed");
        typename std::initializer_list<const Matrix_Map*>::const_iterator source = sources.begin();
        std::size_t i = 0;
        for(Input *input : inputs)
        {
            input->load_layout(stage.inputs[i], **source);
            ++source;
            ++i;
        }
    }

    static void save_dims(Stage &stage, const Dim_mnk &dim0, const Dim_mnk *dim1)
    {
        stage.m0 = dim0.m.shapes;
        stage.n0 = dim0.n.shapes;
        stage.k0 = dim0.k.shapes;
        if(dim1)
        {
            stage.m1 = dim1->m.shapes;
            stage.n1 = dim1->n.shapes;
            stage.k1 = dim1->k.shapes;
        }
    }

    static void load_dims(const Stage &stage, Dim_mnk &dim0, Dim_mnk *dim1)
    {
        dim0.m.shapes = stage.m0;
        dim0.n.shapes = stage.n0;
        dim0.k.shapes = stage.k0;
        if(dim1)
        {
            dim1->m.shapes = stage.m1;
            dim1->n.shapes = stage.n1;
            dim1->k.shapes = stage.k1;
        }
    }

    void save_first(std::initializer_list<Input*> inputs, const Mul &mul,
                    const Tmp *tmp, const Dim_mnk &dim0, const Dim_mnk *dim1)
    {
        save_inputs(first, inputs);
        first.mul_map = mul.ptrList;
        first.mul_first = mul.h_array_1;
        first.mul_size = mul.totalSize;
        if(tmp)
        {
            first.tmp_array = tmp->h_array;
            first.tmp_size = tmp->totalSize;
        }
        save_dims(first, dim0, dim1);
        first.ready = true;
    }

    void load_first(std::initializer_list<Input*> inputs,
                    std::initializer_list<const Matrix_Map*> sources,
                    Mul &mul, Tmp *tmp, Dim_mnk &dim0, Dim_mnk *dim1) const
    {
        load_inputs(first, inputs, sources);
        mul.ptrList = first.mul_map;
        mul.h_array_1 = first.mul_first;
        mul.totalSize = first.mul_size;
        if(tmp)
        {
            tmp->h_array = first.tmp_array;
            tmp->totalSize = first.tmp_size;
        }
        load_dims(first, dim0, dim1);
    }

    void save_second(std::initializer_list<Input*> inputs, const Mul &mul,
                     const Tmp *tmp, const Output &output,
                     const Dim_mnk &dim0, const Dim_mnk *dim1)
    {
        save_inputs(second, inputs);
        second.mul_second = mul.h_array_2;
        if(tmp)
        {
            second.tmp_array = tmp->h_array;
            second.tmp_size = tmp->totalSize;
        }
        second.output_map = output.ptrList;
        second.output_array = output.h_array;
        second.output_size = output.totalSize;
        save_dims(second, dim0, dim1);
        second.ready = true;
    }

    void load_second(std::initializer_list<Input*> inputs,
                     std::initializer_list<const Matrix_Map*> sources,
                     Mul &mul, Tmp *tmp, Output &output,
                     Dim_mnk &dim0, Dim_mnk *dim1) const
    {
        load_inputs(second, inputs, sources);
        mul.h_array_2 = second.mul_second;
        if(tmp)
        {
            tmp->h_array = second.tmp_array;
            tmp->totalSize = second.tmp_size;
        }
        output.ptrList = second.output_map;
        output.h_array = second.output_array;
        output.totalSize = second.output_size;
        load_dims(second, dim0, dim1);
    }
};

template<typename TA, typename TC, typename Tdata>
struct Gpu_LR_Plan_Cache
{
    using TAC = std::pair<TA, TC>;
    using Data_Pool = std::map<std::string, Data_Pack<TA, TC, Tdata>>;
    using List_Map = std::unordered_map<Label::Aab_Aab, List_A<TA, TAC>>;
    using Shape_Map = std::map<TA, std::map<TAC, Shape_Vector>>;

    TC period;
    int threads = 0;
    std::unordered_map<Label::ab, std::string> data_names;
    List_Map lists;
    std::map<std::string, Shape_Map> shapes;
    std::map<Label::ab_ab, Gpu_LR_Case_Plan<TA, TAC, Tdata>> cases;

    static bool same_shape(const Shape_Vector &a, const Shape_Vector &b)
    {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    }

    // Reject reuse when screening, geometry distribution, or block shapes change.
    bool matches(const TC &current_period, int current_threads,
                 const std::unordered_map<Label::ab, std::string> &current_names,
                 const List_Map &current_lists, const Data_Pool &pool) const
    {
        if(period != current_period || threads != current_threads ||
           data_names != current_names || lists.size() != current_lists.size() ||
           shapes.size() != pool.size()) return false;
        for(const auto &entry : current_lists)
        {
            const auto old = lists.find(entry.first);
            if(old == lists.end()) return false;
            const List_A<TA,TAC> &a = old->second;
            const List_A<TA,TAC> &b = entry.second;
            if(a.a01 != b.a01 || a.a2 != b.a2 || a.b01 != b.b01 || a.b2 != b.b2)
                return false;
        }
        for(const auto &named : pool)
        {
            const auto old_name = shapes.find(named.first);
            if(old_name == shapes.end()) return false;
            const Shape_Map &old_map = old_name->second;
            const auto &current_map = named.second.Ds_ab;
            if(old_map.size() != current_map.size()) return false;
            for(const auto &row : current_map)
            {
                const auto old_row = old_map.find(row.first);
                if(old_row == old_map.end() || old_row->second.size() != row.second.size())
                    return false;
                for(const auto &block : row.second)
                {
                    const auto old_block = old_row->second.find(block.first);
                    if(old_block == old_row->second.end() ||
                       !same_shape(old_block->second, block.second.shape)) return false;
                }
            }
        }
        return true;
    }

    void capture(const TC &current_period, int current_threads,
                 const std::unordered_map<Label::ab, std::string> &current_names,
                 const List_Map &current_lists, const Data_Pool &pool)
    {
        period = current_period;
        threads = current_threads;
        data_names = current_names;
        lists = current_lists;
        shapes.clear();
        cases.clear();
        for(const auto &named : pool)
        {
            Shape_Map &named_shapes = shapes[named.first];
            for(const auto &row : named.second.Ds_ab)
                for(const auto &block : row.second)
                    named_shapes[row.first][block.first] = block.second.shape;
        }
    }
};

}

#endif
#endif
