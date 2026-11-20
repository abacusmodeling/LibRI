#ifndef RI_GPU_RESIDENT_OPTIONS_H
#define RI_GPU_RESIDENT_OPTIONS_H

#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace RI
{

// Named geometry tensors share one device budget within an LRI resident cache.
struct GPU_Resident_Options
{
    bool enabled = false;
    std::set<std::string> data_names;
    // 0 selects the automatic budget; positive values specify GiB (1024^3 bytes).
    double budget_gib = 0.0;
};

// Validate before changing live caches; zero remains the automatic-budget sentinel.
inline std::size_t gpu_resident_budget_bytes(double budget_gib)
{
    if(!std::isfinite(budget_gib) || budget_gib < 0.0)
        throw std::invalid_argument("GPU resident budget must be finite and nonnegative (GiB)");
    if(budget_gib == 0.0) return 0;
    const long double bytes = static_cast<long double>(budget_gib) * 1024 * 1024 * 1024;
    if(bytes < 1 || bytes >= static_cast<long double>(std::numeric_limits<std::size_t>::max()))
        throw std::invalid_argument("GPU resident budget is outside the representable byte range");
    return static_cast<std::size_t>(bytes);
}

}

#endif
