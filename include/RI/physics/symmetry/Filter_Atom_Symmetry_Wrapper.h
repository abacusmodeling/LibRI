// ===================
//  Author: Peize Lin
//  date: 2026.11.10
// ===================

#pragma once

#include "Filter_Atom_Symmetry.h"
#include "../../ri/LRI.h"
#ifdef __GPU_RI
#include "Filter_Atom_Symmetry_GPU.h"
#endif
#include <memory>

namespace RI
{

// Own both policies and select once before entering the contraction loops.
// This callable is not an atom filter and does not forward filter callbacks.
template<typename TA, typename TC, typename Tdata>
class Filter_Atom_Symmetry_Wrapper
{
    using TAC = std::pair<TA, TC>;
    struct Configuration
    {
        TC period;
        std::map<std::pair<TA,TA>, std::set<TC>> sector;
    };
    // Shared immutable configuration keeps filter references valid after copies.
    std::shared_ptr<const Configuration> configuration;
    Filter_Atom_Symmetry<TA,TC,Tdata> filter_CPU;
#ifdef __GPU_RI
    Filter_Atom_Symmetry_GPU<TA,TC,Tdata> filter_GPU;
#endif

  public:
    Filter_Atom_Symmetry_Wrapper(
        const TC &period,
        const std::map<std::pair<TA,TA>, std::set<TC>> &irsec)
        : configuration(std::make_shared<Configuration>(Configuration{period, irsec})),
          filter_CPU(configuration->period, configuration->sector)
#ifdef __GPU_RI
        , filter_GPU(configuration->period, configuration->sector)
#endif
    {}

    const Filter_Atom<TA,TAC>* operator()(LRI_Cal_Mode mode) const
    {
        switch (mode)
        {
            case LRI_Cal_Mode::CPU:
            case LRI_Cal_Mode::CPU_fine_grained_lock:
                return &filter_CPU;
#ifdef __GPU_RI
            case LRI_Cal_Mode::GPU:
                return &filter_GPU;
#endif
            default:
                throw std::invalid_argument("Unknown contraction mode for symmetry filter");
        }
    }
};

}
