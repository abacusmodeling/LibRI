#include <RI/ri/gpu/GPU_Resident_Options.h>

#include <iostream>
#include <limits>

int main()
{
    const RI::GPU_Resident_Options defaults;
    if(defaults.enabled || !defaults.data_names.empty() || defaults.budget_gib != 0.0)
        return 1;
    if(RI::gpu_resident_budget_bytes(0.0) != 0
       || RI::gpu_resident_budget_bytes(0.125) != std::size_t(128) * 1024 * 1024
       || RI::gpu_resident_budget_bytes(8.0) != std::size_t(8) * 1024 * 1024 * 1024)
        return 1;
    for(double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::denorm_min()})
    {
        bool rejected = false;
        try { RI::gpu_resident_budget_bytes(invalid); }
        catch(const std::invalid_argument&) { rejected = true; }
        if(!rejected) return 1;
    }
    std::cout << "PASS resident defaults, GiB conversion and invalid budgets\n";
}
