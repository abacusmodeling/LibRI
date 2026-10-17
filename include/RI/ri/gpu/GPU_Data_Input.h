// =================================
//  Author: Laiyuan Yang, Peize Lin
//  date: 2026.01.01
// =================================

#pragma once

#ifdef __GPU_RI

#include "../../global/gpu/GPU_Backend.h"
#include "GPU_Data_Pack.h"

#include <algorithm>
#include <omp.h>
#include <type_traits>

namespace RI
{

namespace GPU_Data
{

template<typename TA, typename TAC, typename Tdata_CPU>
class Input
{
  public:
	GPU_Backend::Queue queue_ = nullptr;
	using Tdata_GPU = GPU_Backend::Type_to_GPU<Tdata_CPU>;
	using Same_data_layout = std::integral_constant<
		bool,
		sizeof(Tdata_CPU)==sizeof(Tdata_GPU) && alignof(Tdata_CPU)==alignof(Tdata_GPU)>;

	Input()
	{
		this->h_data.resize(omp_get_max_threads());
		this->tensor_insert.resize(omp_get_max_threads(), false);
	}

	// Assign each distinct block a stable host-buffer offset before copying data.
	// The caller registers batches serially, then copies distinct blocks in parallel.
	void defer_input_copies()
	{
		this->defer_copies = true;
		this->planned_sizes.assign(this->h_data.size(), 0);
		this->copy_tasks.resize(this->h_data.size());
	}

	void finish_deferred_copies()
	{
		#pragma omp parallel for schedule(static)
		for(int tid=0; tid<static_cast<int>(this->h_data.size()); ++tid)
		{
			std::vector<Tdata_GPU> &buffer = this->h_data[tid];
			buffer.resize(this->planned_sizes[tid]);
			for(const Copy_Task &task : this->copy_tasks[tid])
				copy_data_impl(buffer.data() + task.pos, *task.tensor, Same_data_layout());
		}
		this->copy_tasks.clear();
	}

	const Pack & insert(const TA &Aa, const TAC &Ab, const Tensor<Tdata_CPU> &tensor)
	{
		Pack &pack = this->ptrList[Aa][Ab];
		if(!pack.exist)
		{
			pack.exist = true;
			const int owner = this->defer_copies
				? static_cast<int>(this->next_owner++ % this->h_data.size())
				: omp_get_thread_num();
			pack.pos = this->defer_copies
				? this->planned_sizes[owner]
				: this->h_data[owner].size();
			pack.shape = tensor.shape;
			pack.thread_num = owner;
			if(this->defer_copies)
			{
				this->copy_tasks[owner].push_back({&tensor, pack.pos});
				this->planned_sizes[owner] += tensor.shape.get_shape_all();
			}
			else
				this->tensor_insert[owner] = true;
		}
		else
		{
			if(!this->defer_copies)
				this->tensor_insert[omp_get_thread_num()] = false;
		}
		this->h_array.push_back(pack);
		return pack;
	}

	void insert_data(const Tensor<Tdata_CPU> &tensor)
	{
		const int thread_num = omp_get_thread_num();
		if(!this->defer_copies && this->tensor_insert[thread_num])
			insert_data_impl(this->h_data[thread_num], tensor, Same_data_layout());
	}

	// 将对应的 C, V, D 放在 d_Cs, d_Vs, d_Ds 上
	void upload(GPU_Backend::Queue queue)
	{
		this->queue_ = queue;
		std::vector<std::size_t> h_data_begin(this->h_data.size()+1, 0);
		for(std::size_t i=1; i<h_data_begin.size(); ++i)
			h_data_begin[i] = h_data_begin[i-1] + this->h_data[i-1].size();

		GPU_Backend::allocate(&this->d_data, h_data_begin.back(), queue);
		for(std::size_t i=0; i<this->h_data.size(); ++i)
			GPU_Backend::upload(
				this->h_data[i].size(), this->h_data[i].data(),
				this->d_data+h_data_begin[i], queue);

		const std::size_t batchCount = this->h_array.size();
		std::vector<Tdata_GPU*> d_array_(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for(std::size_t i=0; i<batchCount; ++i)
			d_array_[i] = this->d_data + this->h_array[i].pos + h_data_begin[this->h_array[i].thread_num];
		GPU_Backend::allocate(&this->d_array, batchCount, queue);
		GPU_Backend::upload(batchCount, d_array_.data(), this->d_array, queue);
	}

	~Input()
	{
		GPU_Backend::free(this->d_data, this->queue_);
		GPU_Backend::free(this->d_array, this->queue_);
	}

  private:
	struct Copy_Task
	{
		const Tensor<Tdata_CPU> *tensor;
		std::size_t pos;
	};

	static void copy_data_impl(Tdata_GPU *dest, const Tensor<Tdata_CPU> &tensor, std::true_type)
	{
		std::copy(tensor.ptr(), tensor.ptr() + tensor.shape.get_shape_all(), dest);
	}

	static void copy_data_impl(Tdata_GPU *dest, const Tensor<Tdata_CPU> &tensor, std::false_type)
	{
		const Tdata_CPU *src = tensor.ptr();
		for(std::size_t i=0; i<tensor.shape.get_shape_all(); ++i)
			dest[i] = GPU_Backend::data_to_GPU(src[i]);
	}

	bool defer_copies = false;
	std::size_t next_owner = 0;
	std::vector<std::size_t> planned_sizes;
	std::vector<std::vector<Copy_Task>> copy_tasks;

	static void insert_data_impl(
		std::vector<Tdata_GPU> &h_data,
		const Tensor<Tdata_CPU> &tensor,
		std::true_type)
	{
		h_data.insert(
			h_data.end(),
			tensor.ptr(),
			tensor.ptr() + tensor.shape.get_shape_all());
	}

	static void insert_data_impl(
		std::vector<Tdata_GPU> &h_data,
		const Tensor<Tdata_CPU> &tensor,
		std::false_type)
	{
		const std::size_t size = tensor.shape.get_shape_all();
		h_data.resize(h_data.size() + size);
		const auto ptr_dest = h_data.end() - size;
		const auto ptr_src = tensor.ptr();
		for(std::size_t i=0; i<size; ++i)
			ptr_dest[i] = GPU_Backend::data_to_GPU(ptr_src[i]);
	}

  public:
	std::vector<std::vector<Tdata_GPU>> h_data;		// 存储数据（CPU）
	Tdata_GPU *d_data = nullptr;					// 存储数据（GPU）
	Tdata_GPU **d_array = nullptr;					// 记录每个batch的 d_data 指针（GPU）
	std::vector<Pack> h_array;					// 记录每个batch的Pack
	std::map<TA, std::map<TAC, Pack>> ptrList;	// 记录每个原子对的Pack
	std::vector<bool> tensor_insert;			// 记录是否当前张量需要insert_data
};

}

}

#endif
