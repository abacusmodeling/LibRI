// =================================
//  Author: Laiyuan Yang, Peize Lin
//  date: 2026.01.01
// =================================

#pragma once

#ifdef __GPU_RI

#include "../../global/gpu/GPU_Backend.h"
#include "GPU_Data_Pack.h"

namespace RI
{

namespace GPU_Data
{

template<typename TA, typename TAC, typename Tdata_CPU>
class Tmp
{
  public:
	using Tdata_GPU = GPU_Backend::Type_to_GPU<Tdata_CPU>;

	const Pack &insert(const Shape_Vector &shape)
	{
		Pack pack;
		pack.exist = true;
		pack.pos = this->totalSize;
		pack.shape = shape;
		this->totalSize += shape.get_shape_all();
		this->h_array.push_back(pack);
		return this->h_array.back();
	}

	std::vector<GPU_Backend::Int> upload(
		const std::size_t memory_limit, GPU_Backend::Queue queue)
	{
		this->queue_ = queue;
		const std::size_t size_limit = std::min(this->totalSize, memory_limit / sizeof(Tdata_GPU));
		const std::vector<std::vector<Pack>> h_array_segments = segment_points(this->h_array, size_limit);

		GPU_Backend::allocate(&this->d_data, size_limit, queue);
		GPU_Backend::memset(
			this->d_data, 0, size_limit * sizeof(Tdata_GPU), queue);

		const std::size_t batchCount = this->h_array.size();
		std::vector<Tdata_GPU*> d_array_;							// 记录每个batch的 d_data 指针（CPU）
		for(const auto &h_array_segment : h_array_segments)
			for(const Pack &pack : h_array_segment)
				d_array_.push_back(this->d_data + pack.pos);
		GPU_Backend::allocate(&this->d_array, batchCount, queue);
		GPU_Backend::upload(batchCount, d_array_.data(), this->d_array, queue);

		std::vector<GPU_Backend::Int> segments_size;
		for(const auto &h_array_segment : h_array_segments)
			segments_size.push_back(h_array_segment.size());
		return segments_size;
	}

	// 将数轴上的点按段长不超过S进行分段
	// 参数：
	//   points: 一维数轴上的点（与零点的距离），vector<double> 类型
	//   size_limit: 每一段允许的最大长度
	// 返回值：嵌套的vector，每个子vector是一段内的点
	std::vector<std::vector<Pack>> segment_points(
		const std::vector<Pack> &points,
		const std::size_t size_limit)
	{
		std::vector<std::vector<Pack>> segments;
		const std::size_t size_all = points.size();
		for (std::size_t start_idx=0; start_idx<size_all; )
		{
			const std::size_t start_pos = points[start_idx].pos;
			std::size_t end_idx = start_idx;
			std::vector<Pack> current_segment;
			for( ; end_idx<size_all && (points[end_idx].pos+points[end_idx].shape.get_shape_all()-start_pos)<=size_limit; ++end_idx )
				current_segment.push_back({true, points[end_idx].pos-start_pos, points[end_idx].shape});
			segments.emplace_back(std::move(current_segment));
			start_idx = end_idx;
		}
		return segments;
	}

	~Tmp()
	{
		GPU_Backend::free(this->d_data, this->queue_);
		GPU_Backend::free(this->d_array, this->queue_);
	}

	std::size_t totalSize = 0;					// 总的数据数量
	Tdata_GPU *d_data = nullptr;					// 存储数据（GPU）
	std::vector<Pack> h_array;					// 记录每个batch的Pack
	Tdata_GPU **d_array = nullptr;					// 记录每个batch的 d_data 指针（GPU）

  private:
	// The context must outlive the buffers allocated on its queue.
	GPU_Backend::Queue queue_ = nullptr;
};

}

}

#endif
