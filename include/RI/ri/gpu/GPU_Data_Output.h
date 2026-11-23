// =================================
//  Author: Laiyuan Yang, Peize Lin
//  date: 2026.01.01
// =================================

#pragma once

#ifdef __GPU_RI

#include "../../global/gpu/GPU_Backend.h"
#include "GPU_Data_Pack.h"
#include "../LRI_Cal_Aux.h"

#include <cstring>
#include <type_traits>
#include <utility>
#include <memory>

namespace RI
{

namespace GPU_Data
{

template<typename TA, typename TAC, typename Tdata_CPU>
class Output : public Pair_Packs<TA,TAC>
{
    Thread_Batch<const Pack*> pending;
    const std::vector<std::size_t> *cached_array = nullptr;
  public:
    GPU_Backend::Queue queue_ = nullptr;
	using Tdata_GPU = GPU_Backend::Type_to_GPU<Tdata_CPU>;
	using Same_data_layout = std::integral_constant<
		bool,
		sizeof(Tdata_CPU)==sizeof(Tdata_GPU) && alignof(Tdata_CPU)==alignof(Tdata_GPU)>;

	const Pack &insert(const TA &Aa, const TAC &Ab, const Shape_Vector &shape)
	{
		Pack &pack = this->local_packs()[Aa][Ab];
		if(!pack.exist)
		{
			pack.exist = true;
			pack.shape = shape;
		}
		pending.push_back(&pack);
		return pack;
	}

    void prepare_layout()
    {
        if(!pending.size()) return;
        this->merge_packs();
        pending.for_each([&](const Pack *p){batch_offsets.push_back(p->pos);});
        pending.release(); this->release_local_packs();
    }
    const std::vector<std::size_t>& offsets() const { return cached_array ? *cached_array : batch_offsets; }
    void bind_layout(const typename Pair_Packs<TA,TAC>::Pack_Map &map, const std::vector<std::size_t> &array)
    { this->bind_pairs(map); cached_array = &array; }
	void upload(GPU_Backend::Queue queue)
	{
        this->queue_=queue;
        prepare_layout();
		GPU_Backend::allocate(&this->d_data, this->totalSize, queue);
		GPU_Backend::memset(
			this->d_data, 0, this->totalSize * sizeof(Tdata_GPU), queue);

		const std::size_t batchCount = offsets().size();
		upload_pointers.clear();
		upload_pointers.reserve(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for (std::size_t i = 0; i < batchCount; i++)
			upload_pointers.push_back(this->d_data + offsets()[i]);
		GPU_Backend::allocate(&this->d_array, batchCount, queue);
		GPU_Backend::upload(batchCount, upload_pointers.data(), this->d_array, queue);
	}

	void download(
		std::map<TA, std::map<TAC, Tensor<Tdata_CPU>>> &Ds_result,
		GPU_Backend::Queue queue,
		const Tdata_CPU fac) const
	{
		// The download overwrites every element before the synchronization/readback.
		// Arithmetic types need no zero pass; class types retain default construction.
		std::unique_ptr<Tdata_GPU[]> h_data(
			this->totalSize ? new Tdata_GPU[this->totalSize] : nullptr);
		GPU_Backend::download(
			this->totalSize, this->d_data, h_data.get(), queue);
		GPU_Backend::sync(queue);
		for(const auto &ptrList_A : this->pairs())
		{
			for(const auto &ptrList_B : ptrList_A.second)
			{
				const Pack &pack = ptrList_B.second;
				Tensor<Tdata_CPU> D_output(pack.shape);
				download_data_impl(D_output, h_data.get(), pack, Same_data_layout());

				Tensor<Tdata_CPU> &D_result = Ds_result[ptrList_A.first][ptrList_B.first];
				LRI_Cal_Aux::add_Ds(std::move(D_output), D_result, fac);
			}
		}
	}

    // Output shapes and offsets in pairs() must survive until download restores AO blocks.
    void release_completed_uploads()
    {
        std::vector<Tdata_GPU*>().swap(upload_pointers);
        if(!cached_array) std::vector<std::size_t>().swap(batch_offsets);
    }
	~Output()
	{
		GPU_Backend::free(this->d_data, this->queue_);
		GPU_Backend::free(this->d_array, this->queue_);
	}

  private:
	static void download_data_impl(
		Tensor<Tdata_CPU> &D_output,
		const Tdata_GPU *h_data,
		const Pack &pack,
		std::true_type)
	{
		std::memcpy(
			D_output.ptr(),
			h_data+pack.pos,
			pack.shape.get_shape_all()*sizeof(Tdata_CPU));
	}

	static void download_data_impl(
		Tensor<Tdata_CPU> &D_output,
		const Tdata_GPU *h_data,
		const Pack &pack,
		std::false_type)
	{
		const std::size_t size = pack.shape.get_shape_all();
		const auto ptr_src = h_data+pack.pos;
		const auto ptr_dest = D_output.ptr();
		for(std::size_t i=0; i<size; ++i)
			ptr_dest[i] = GPU_Backend::data_to_CPU(ptr_src[i]);
	}

  public:
    std::vector<Tdata_GPU*> upload_pointers;
	Tdata_GPU *d_data = nullptr;					// 存储数据（GPU）
	std::vector<std::size_t> batch_offsets;					// One element offset per contribution; output shapes stay in the unique-pair map.
	Tdata_GPU **d_array = nullptr;					// 记录每个batch的 d_data 指针（GPU）
};

}

}

#endif
