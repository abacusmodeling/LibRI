// =================================
//  Author: Laiyuan Yang, Peize Lin
//  date: 2026.01.01
// =================================

#pragma once

#ifdef __GPU_RI

#include "../../global/gpu/GPU_Backend.h"
#include "GPU_Data_Pack.h"
#include <stdexcept>

namespace RI
{

namespace GPU_Data
{

template<typename TA, typename TAC, typename Tdata_CPU>
class Tmp
{
    Thread_Batch<std::size_t> pending;
    const std::vector<std::size_t> *cached_sizes = nullptr;
  public:
    GPU_Backend::Queue queue_ = nullptr;
    using Tdata_GPU = GPU_Backend::Type_to_GPU<Tdata_CPU>;

    // Shapes are used immediately by the caller; only the element count persists.
    Pack insert(const Shape_Vector &shape)
    {
        Pack pack; pack.exist=true; pack.shape=shape;
        pending.push_back(shape.get_shape_all()); return pack;
    }
    void prepare_layout()
    {
        pending.for_each([&](std::size_t size){totalSize+=size;h_array.push_back(size);});
        pending.release();
    }
    const std::vector<std::size_t>& element_counts() const { return cached_sizes ? *cached_sizes : h_array; }
    void bind_layout(const std::vector<std::size_t> &sizes) { cached_sizes = &sizes; }
    std::vector<GPU_Backend::Int> upload(std::size_t memory_limit,GPU_Backend::Queue queue)
    {
        this->queue_=queue;
        prepare_layout();
        const std::size_t limit=std::min(totalSize,memory_limit/sizeof(Tdata_GPU));
        for(auto size:element_counts())
            if(size>limit) throw std::length_error("LibRI temporary tensor exceeds segment capacity");
        GPU_Backend::allocate(&d_data,limit, queue);
        GPU_Backend::memset(d_data,0,limit*sizeof(Tdata_GPU),queue);
        upload_pointers.reserve(element_counts().size());
        std::vector<GPU_Backend::Int> segments;
        std::size_t used=0;
        GPU_Backend::Int count=0;
        for(auto size:element_counts())
        {
            if(count && size>limit-used) { segments.push_back(count); used=0; count=0; }
            upload_pointers.push_back(d_data+used);
            used+=size; ++count;
        }
        if(count) segments.push_back(count);
        GPU_Backend::allocate(&d_array,upload_pointers.size(), queue);
        GPU_Backend::upload(upload_pointers.size(),upload_pointers.data(),d_array,queue);
        return segments;
    }

    void release_completed_uploads()
    {
        std::vector<Tdata_GPU*>().swap(upload_pointers);
        if(!cached_sizes) std::vector<std::size_t>().swap(h_array);
    }
	~Tmp()
	{
		GPU_Backend::free(this->d_data, this->queue_);
		GPU_Backend::free(this->d_array, this->queue_);
	}

	std::size_t totalSize = 0;					// 总的数据数量
    std::vector<Tdata_GPU*> upload_pointers;
	Tdata_GPU *d_data = nullptr;					// 存储数据（GPU）
	std::vector<std::size_t> h_array;					// 记录每个batch的Pack
	Tdata_GPU **d_array = nullptr;					// 记录每个batch的 d_data 指针（GPU）
};

}

}

#endif
