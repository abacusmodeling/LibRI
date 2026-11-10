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
class Mul
{
  public:
	GPU_Backend::Queue queue_ = nullptr;
	using Tdata_GPU = GPU_Backend::Type_to_GPU<Tdata_CPU>;

	const Pack &insert_1st(const TA &Aa, const TAC &Ab, const Shape_Vector &shape)
	{
		Pack &pack = this->ptrList[Aa][Ab];
		if(!pack.exist)
		{
			pack.exist = true;
			pack.pos = this->totalSize;
			pack.shape = shape;
			this->totalSize += shape.get_shape_all();
		}
		this->h_array_1.push_back(pack);
		return pack;
	}

	void upload_1st(GPU_Backend::Queue queue)
	{
		this->queue_ = queue;
		GPU_Backend::allocate(&this->d_data, this->totalSize, queue);
		GPU_Backend::memset(
			this->d_data, 0, totalSize * sizeof(Tdata_GPU), queue);

		const std::size_t batchCount = this->h_array_1.size();
		std::vector<Tdata_GPU*> d_array_(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for (std::size_t i = 0; i < batchCount; i++)
			d_array_[i] = this->d_data + this->h_array_1[i].pos;
		GPU_Backend::allocate(&this->d_array_1, batchCount, queue);
		GPU_Backend::upload(
			batchCount, d_array_.data(), this->d_array_1, queue);
	}

	const Pack &find_2nd(const TA &Aa, const TAC &Ab)
	{
		return Global_Func::find(this->ptrList, Aa, Ab);
	}

	const Pack &insert_2nd(const Pack &pack)
	{
		this->h_array_2.push_back(pack);
		return this->h_array_2.back();
	}

	// 在 GPU 上分配空间，并获取指针
	void upload_2nd(GPU_Backend::Queue queue)
	{
		this->queue_ = queue;
		const std::size_t batchCount = this->h_array_2.size();
		std::vector<Tdata_GPU*> d_array_2_(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for (std::size_t i = 0; i < batchCount; i++)
			d_array_2_[i] = this->d_data + this->h_array_2[i].pos;
		GPU_Backend::allocate(&this->d_array_2, batchCount, queue);
		GPU_Backend::upload(
			batchCount, d_array_2_.data(), this->d_array_2, queue);
	}

	~Mul()
	{
		GPU_Backend::free(this->d_data, this->queue_);
		GPU_Backend::free(this->d_array_1, this->queue_);
		GPU_Backend::free(this->d_array_2, this->queue_);
	}

	std::size_t totalSize = 0;                  // 总的数据数量
	Tdata_GPU *d_data = nullptr;					// 存储数据（GPU）
	std::vector<Pack> h_array_1;				// 记录每个batch的Pack
	std::vector<Pack> h_array_2;				// 记录每个batch的Pack
	Tdata_GPU **d_array_1 = nullptr;				// 记录每个batch的 d_data 指针（GPU）
	Tdata_GPU **d_array_2 = nullptr;				// 记录每个batch的 d_data 指针（GPU）
	std::map<TA, std::map<TAC, Pack>> ptrList;	// 记录每个原子对的Pack
};

}

}

#endif
