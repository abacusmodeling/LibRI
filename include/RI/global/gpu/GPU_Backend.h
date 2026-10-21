// ===================
//  Author: Haobo Chen
//  date: 2026.07.21
// ===================

#pragma once

//  LibRI GPU backend adapter

#if (defined(__MAGMA_RI) + defined(__DDLA_RI)) != 1
    #error "Define exactly one of __MAGMA_RI or __DDLA_RI"
#endif

#ifdef __DDLA_RI
    #if (defined(DDLA_USE_CUDA) + defined(DDLA_USE_HIP)) != 1
        #error "__DDLA_RI requires exactly one of DDLA_USE_CUDA or DDLA_USE_HIP"
    #endif
    #include <ddla/ddla_connector.h>
    #include <ddla/ddla_handle_t.h>
    #include <ddla/gemmVbatched.h>

#elif defined(__MAGMA_RI)
    #include "Magmablas_Interface-Contiguous.h"
    #include "Magma_Wrapper.h"
    #include <magma_v2.h>
    #define MAGMA_CHECK_RI(x) if((x)!=MAGMA_SUCCESS)	throw std::runtime_error(std::string(__FILE__)+" line "+std::to_string(__LINE__)+" error "+std::string(magma_strerror(x)));
#endif

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace RI
{
namespace GPU_Backend
{

#ifdef __DDLA_RI
    using Int = int;
    using Transpose = ddla::deblasOperation_t;
    using Queue = ddla::DdlaHandle_t;
    constexpr Transpose NoTrans = ddla::DEBLAS_OP_N;
    constexpr Transpose Trans = ddla::DEBLAS_OP_T;
    constexpr Transpose ConjTrans = ddla::DEBLAS_OP_C;

    inline ddla::deviceStream_t stream(Queue queue)
    {
        return static_cast<ddla::deviceStream_t>(ddla::ddla_get_stream(queue));
    }
#elif defined(__MAGMA_RI)
    using Int = magma_int_t;
    using Transpose = magma_trans_t;
    using Queue = magma_queue_t;
    constexpr Transpose NoTrans = MagmaNoTrans;
    constexpr Transpose Trans = MagmaTrans;
    constexpr Transpose ConjTrans = MagmaConjTrans;
#endif

template<typename T_CPU> struct Type_to_GPU_Helper { using T_GPU = T_CPU; };
#ifdef __MAGMA_RI
    template<> struct Type_to_GPU_Helper<std::complex<float>> { using T_GPU = magmaFloatComplex; };
    template<> struct Type_to_GPU_Helper<std::complex<double>> { using T_GPU = magmaDoubleComplex; };
#endif
template <typename T_CPU> using Type_to_GPU = typename Type_to_GPU_Helper<T_CPU>::T_GPU;

template<typename T_GPU> struct Type_to_CPU_Helper { using T_CPU = T_GPU; };
#ifdef __MAGMA_RI
    template<> struct Type_to_CPU_Helper<magmaFloatComplex> { using T_CPU = std::complex<float>; };
    template<> struct Type_to_CPU_Helper<magmaDoubleComplex> { using T_CPU = std::complex<double>; };
#endif
template <typename T_GPU> using Type_to_CPU = typename Type_to_CPU_Helper<T_GPU>::T_CPU;

template<typename T_CPU> Type_to_GPU<T_CPU> data_to_GPU(const T_CPU data_cpu)
{
    (void)data_cpu; // mark as used to suppress the unused-parameter warning; the guard intentionally always throws
    throw std::invalid_argument(std::string(__FILE__) + " line " + std::to_string(__LINE__) + ": data_to_GPU should not be used");
}
template<typename T_GPU> Type_to_CPU<T_GPU> data_to_CPU(const T_GPU data_gpu)
{
    (void)data_gpu; // mark as used to suppress the unused-parameter warning; the guard intentionally always throws
    throw std::invalid_argument(std::string(__FILE__) + " line " + std::to_string(__LINE__) + ": data_to_CPU should not be used");
}
#ifdef __MAGMA_RI
    template<> magmaFloatComplex    data_to_GPU(const std::complex<float>  data_cpu) { return {std::real(data_cpu), std::imag(data_cpu)}; }
    template<> magmaDoubleComplex   data_to_GPU(const std::complex<double> data_cpu) { return {std::real(data_cpu), std::imag(data_cpu)}; }
    template<> std::complex<float>  data_to_CPU(const magmaFloatComplex    data_gpu) { return {MAGMA_C_REAL(data_gpu), MAGMA_C_IMAG(data_gpu)}; }
    template<> std::complex<double> data_to_CPU(const magmaDoubleComplex   data_gpu) { return {MAGMA_C_REAL(data_gpu), MAGMA_C_IMAG(data_gpu)}; }
#endif


inline void validate_local_gpu_count(const MPI_Comm& communicator)
{
    MPI_Comm local_communicator = MPI_COMM_NULL;
    if (MPI_Comm_split_type(
            communicator, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
            &local_communicator)
        != MPI_SUCCESS)
        throw std::runtime_error("failed to create the LibRI node communicator");

    int local_size = 0;
    MPI_Comm_size(local_communicator, &local_size);
    MPI_Comm_free(&local_communicator);

    int device_count = 0;
#ifdef __DDLA_RI
    ddla::DEVICE_CHECK(ddla::deviceGetDeviceCount(&device_count));
#elif defined(__MAGMA_RI)
    device_count = Magma_Wrapper::magma_get_size();
#endif
    if (device_count <= 0 || local_size > device_count)
        throw std::runtime_error(
            "LibRI requires at least one GPU per local MPI rank");
}

class Context
{
  public:
    explicit Context(const MPI_Comm& communicator)
    {
#ifdef __DDLA_RI
        validate_local_gpu_count(communicator);
        ddla::ddla_init(queue_);
        ddla::ddla_set(queue_, communicator);
#elif defined(__MAGMA_RI)
        MAGMA_CHECK_RI(magma_init());
        try
        {
            validate_local_gpu_count(communicator);
        }
        catch (...)
        {
            MAGMA_CHECK_RI(magma_finalize());
            throw;
        }
        int rank = 0;
        MPI_Comm_rank(communicator, &rank);
        magma_setdevice(rank);
        magma_queue_create(rank, &queue_);
#endif
    }

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    ~Context()
    {
#ifdef __DDLA_RI
        ddla::ddla_destroy(queue_);
#elif defined(__MAGMA_RI)
        if (queue_ != nullptr)
            magma_queue_destroy(queue_);
        magma_finalize();
#endif
    }

    Queue& queue()
    {
        return queue_;
    }

    Queue queue_ = nullptr;
};

template <typename T>
inline void allocate(T** pointer, std::size_t count)
{
    if (count == 0)
    {
        *pointer = nullptr;
        return;
    }
#ifdef __DDLA_RI
    ddla::DEVICE_CHECK(ddla::deviceMalloc(pointer, count * sizeof(T)));
#elif defined(__MAGMA_RI)
    MAGMA_CHECK_RI(magma_malloc(reinterpret_cast<void**>(pointer), count * sizeof(T)));
#endif
}

template <typename T>
inline void free(T* pointer)
{
    if (pointer == nullptr)
        return;
#ifdef __DDLA_RI
    ddla::DEVICE_CHECK(ddla::deviceFree(pointer));
#elif defined(__MAGMA_RI)
    MAGMA_CHECK_RI(magma_free(pointer));
#endif
}

template <typename T>
inline void upload(
    std::size_t count, const T* host_data, T* device_data, Queue queue)
{
    if (count == 0)
        return;
#ifdef __DDLA_RI
    ddla::DEVICE_CHECK(deviceMemcpyAsync(
        device_data, host_data, count * sizeof(T),
        ddla::deviceMemcpyHostToDevice, stream(queue)));
#elif defined(__MAGMA_RI)
    magma_setvector_async(
        count, sizeof(T), host_data, 1, device_data, 1, queue);
#endif
}

template <typename T>
inline void download(
    std::size_t count, const T* device_data, T* host_data, Queue queue)
{
    if (count == 0)
        return;
#ifdef __DDLA_RI
    ddla::DEVICE_CHECK(deviceMemcpyAsync(
        host_data, device_data, count * sizeof(T),
        ddla::deviceMemcpyDeviceToHost, stream(queue)));
#elif defined(__MAGMA_RI)
    magma_getvector_async(
        count, sizeof(T), device_data, 1, host_data, 1, queue);
#endif
}

inline void memset(void* pointer, int value, std::size_t bytes, Queue queue)
{
    if (bytes == 0)
        return;
#ifdef __DDLA_RI
    ddla::DEVICE_CHECK(deviceMemsetAsync(
        pointer, value, bytes, stream(queue)));
#elif defined(__MAGMA_RI)
    MAGMA_CHECK_RI(magma_memset(pointer, value, bytes));
#endif
}

inline void sync(Queue queue)
{
#ifdef __DDLA_RI
    ddla::DEVICE_CHECK(
        ddla::deviceStreamSynchronize(stream(queue)));
#elif defined(__MAGMA_RI)
    magma_queue_sync(queue);
#endif
}

template <typename T>
inline void gemmVbatched(
    Transpose transA, Transpose transB,
    Int* m, Int* n, Int* k,
    T alpha, const T* const* dA_array, const T* const* dB_array,
    T beta, T** dC_array, Int batch_count, Queue queue)
{
#ifdef __DDLA_RI
    Int*& lda = transA == NoTrans ? k : m;
    Int*& ldb = transB == NoTrans ? n : k;
    Int*& ldc = n;
    ddla::gemmVbatched(
        transB, transA, n, m, k,
        alpha, dB_array, ldb, dA_array, lda,
        beta, dC_array, ldc, batch_count, queue);
#elif defined(__MAGMA_RI)
    Magma_Interface::gemm_vbatched(
        transA, transB, m, n, k,
        alpha, dA_array, dB_array, beta, dC_array,
        batch_count, queue);
#endif
}

template <typename T>
inline void gemmVbatched2s(
    Transpose transA_0, Transpose transB_0,
    Int* m_0, Int* n_0, Int* k_0,
    T alpha_0, const T* const* dA_array_0,
    const T* const* dB_array_0,
    T beta_0, T** dC_array_0,
    Transpose transA_1, Transpose transB_1,
    Int* m_1, Int* n_1, Int* k_1,
    T alpha_1, const T* const* dAB_array_1,
    T beta_1, T** dC_array_1,
    bool C0_left, Int batch_count,
    const std::vector<Int>& segment_sizes, Queue queue)
{
#ifdef __DDLA_RI
    Int* lda_0 = transA_0 == NoTrans ? k_0 : m_0;
    Int* ldb_0 = transB_0 == NoTrans ? n_0 : k_0;
    Int* ldc_0 = n_0;
    Int* lda_1 = transA_1 == NoTrans ? k_1 : m_1;
    Int* ldb_1 = transB_1 == NoTrans ? n_1 : k_1;
    Int* ldc_1 = n_1;
    ddla::gemmVbatched2s(
        transB_0, transA_0, n_0, m_0, k_0,
        alpha_0, dB_array_0, ldb_0, dA_array_0, lda_0,
        beta_0, dC_array_0, ldc_0,
        transB_1, transA_1, n_1, m_1, k_1,
        alpha_1, dAB_array_1, ldb_1, lda_1,
        beta_1, dC_array_1, ldc_1, !C0_left,
        batch_count, segment_sizes.data(), segment_sizes.size(), queue);
#elif defined(__MAGMA_RI)
    Magma_Interface::gemm_vbatched_2s(
        transA_0, transB_0, m_0, n_0, k_0,
        alpha_0, dA_array_0, dB_array_0, beta_0, dC_array_0,
        transA_1, transB_1, m_1, n_1, k_1,
        alpha_1, dAB_array_1, beta_1, dC_array_1,
        C0_left, batch_count, segment_sizes.data(), queue);
#endif
}

} // namespace GPU_Backend
} // namespace RI

#ifdef __MAGMA_RI
#undef MAGMA_CHECK_RI
#endif