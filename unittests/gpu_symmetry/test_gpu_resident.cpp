#include "tensor_test_helpers.h"
#ifdef __GPU_RI
#include <cuda_runtime_api.h>
#endif

using namespace tensor_test;
using Input = RI::Label::ab;
template <typename T> using Lri = RI::LRI<int, int, 1, T>;

// Register only tensors required by this contraction; missing unrelated inputs are valid.
std::vector<Input> inputs(Label label)
{
    std::vector<Input> result{Input::a, Input::b};
    const auto name = RI::Label_Tools::get_name(label);
    for (const auto input : RI::Label::array_ab)
        if (input != Input::a && input != Input::b
            && name.find(RI::Label_Tools::get_name(input)) != std::string::npos)
            result.push_back(input);
    return result;
}

template <typename T>
void set_input(Lri<T>& lri, Input input, int version, T imaginary)
{
    const bool coefficient = input == Input::a || input == Input::b;
    const RI::Shape_Vector shape = coefficient ? RI::Shape_Vector{2, 2, 2} : RI::Shape_Vector{2, 2};
    Tensors<T> tensors;
    for (int left = 0; left < 2; ++left)
        for (int right = 0; right < 2; ++right)
            for (int cell = 0; cell < 3; ++cell)
                tensors[left][{right, {cell}}] = make_tensor(shape, left, right, cell, version, imaginary);
    lri.set_tensors_map2(tensors, {input});
}

template <typename T>
Lri<T> make_lri(Label label, bool resident, T imaginary)
{
    Lri<T> lri;
    lri.set_parallel(MPI_COMM_WORLD, {{0, {0.0}}, {1, {0.4}}}, {{{{1.0}}}}, Cell{{3}}, {label});
    for (const auto input : inputs(label))
        set_input(lri, input, 0, imaginary);
#ifdef __GPU_RI
    if (resident)
        lri.set_gpu_resident_data_names({"a", "b", "a0b0"});
#endif
    return lri;
}

template <typename T>
void verify(Lri<T>& lri, Label label, bool gpu, const std::string& context)
{
    Tensors<T> expected, actual;
    lri.cal_mode = RI::LRI_Cal_Mode::CPU;
    lri.cal_loop3({label}, expected);
    lri.cal_mode = RI::LRI_Cal_Mode::CPU_fine_grained_lock;
#ifdef __GPU_RI
    if (gpu) lri.cal_mode = RI::LRI_Cal_Mode::GPU;
#endif
    lri.cal_loop3({label}, actual);
    compare(actual, expected, context);
}

template <typename T>
void check_resident(T imaginary, bool gpu)
{
    for (const auto label : RI::Label::array_ab_ab)
        for (bool resident : {false, true})
        {
            auto lri = make_lri(label, resident, imaginary);
            verify(lri, label, gpu, "minimal inputs " + RI::Label_Tools::get_name(label));
        }

    const auto label = Label::a0b0_a1b1;
    // Both copying APIs must preserve the source's warmed C/V after a same-shaped update.
    for (bool assignment : {false, true})
        for (auto changed : {Input::a, Input::b, Input::a0b0, Input::a1b1})
            for (bool free_first : {false, true})
            {
                auto original = make_lri(label, true, imaginary);
                verify(original, label, gpu, "warm original");
                auto copy = assignment ? Lri<T>() : Lri<T>(original);
                if (assignment) copy = original;
                if (free_first)
                    copy.free_tensors_map2(RI::Label_Tools::get_name(changed));
                set_input(copy, changed, 19, imaginary);
                verify(copy, label, gpu, "changed copy");
                verify(original, label, gpu, "original after copy update");
                verify(copy, label, gpu, "changed copy after original reuse");
            }

    auto original = make_lri(label, true, imaginary);
    verify(original, label, gpu, "warm asymmetric owners");
    std::unique_ptr<Lri<T>> rank_zero_copy;
    if (RI::MPI_Wrapper::mpi_get_rank(MPI_COMM_WORLD) == 0)
        rank_zero_copy.reset(new Lri<T>(original));
    // Only rank 0 detaches. Constructing a new MPI/GPU context here would deadlock.
    set_input(original, Input::a, 31, imaginary);
    verify(original, label, gpu, "rank-dependent cache owner counts");
}

int main(int argc, char** argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    const bool gpu = argc > 1 && std::string(argv[1]) == "gpu";
    try
    {
        if (provided < MPI_THREAD_MULTIPLE)
            throw std::runtime_error("MPI_THREAD_MULTIPLE required");
        if (gpu)
        {
            int available = 0;
#ifdef __GPU_RI
            int count = 0;
            available = cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#endif
            int all_available = 0;
            MPI_Allreduce(&available, &all_available, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (!all_available)
            {
                std::cout << "SKIP: CUDA device unavailable\n";
                MPI_Finalize();
                return 77;
            }
        }
        check_resident<double>(0.0, gpu);
        check_resident<std::complex<double>>({0.0, 1.0}, gpu);
        std::cout << "PASS minimal inputs and resident copy isolation (" << (gpu ? "GPU" : "CPU reference") << ")\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
}
