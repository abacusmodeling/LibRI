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

namespace RI
{

namespace GPU_Data
{

template<typename TA, typename TAC, typename Tdata_CPU>
class Output
{
  public:
	using Tdata_GPU = GPU_Backend::Type_to_GPU<Tdata_CPU>;
	using Same_data_layout = std::integral_constant<
		bool,
		sizeof(Tdata_CPU)==sizeof(Tdata_GPU) && alignof(Tdata_CPU)==alignof(Tdata_GPU)>;

	const Pack &insert(const TA &Aa, const TAC &Ab, const Shape_Vector &shape)
	{
		Pack &pack = this->ptrList[Aa][Ab];
		if(!pack.exist)
		{
			pack.exist = true;
			pack.pos = this->totalSize;
			pack.shape = shape;
			this->totalSize += shape.get_shape_all();
		}
		this->h_array.push_back(pack);
		return pack;
	}

	void upload(GPU_Backend::Queue queue)
	{
		GPU_Backend::allocate(&this->d_data, this->totalSize);
		GPU_Backend::memset(
			this->d_data, 0, totalSize * sizeof(Tdata_GPU), queue);

		const std::size_t batchCount = this->h_array.size();
		std::vector<Tdata_GPU*> d_array_(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for (std::size_t i = 0; i < batchCount; i++)
			d_array_[i] = this->d_data + this->h_array[i].pos;
		GPU_Backend::allocate(&this->d_array, batchCount);
		GPU_Backend::upload(batchCount, d_array_.data(), this->d_array, queue);
	}

	void download(
		std::map<TA, std::map<TAC, Tensor<Tdata_CPU>>> &Ds_result,
		GPU_Backend::Queue queue,
		const Tdata_CPU fac) const
	{
		std::vector<Tdata_GPU> h_data(this->totalSize);
		GPU_Backend::download(
			this->totalSize, this->d_data, h_data.data(), queue);
		GPU_Backend::sync(queue);
		for(const auto &ptrList_A : this->ptrList)
		{
			for(const auto &ptrList_B : ptrList_A.second)
			{
				const Pack &pack = ptrList_B.second;
				Tensor<Tdata_CPU> D_output(pack.shape);
				download_data_impl(D_output, h_data, pack, Same_data_layout());

				Tensor<Tdata_CPU> &D_result = Ds_result[ptrList_A.first][ptrList_B.first];
				LRI_Cal_Aux::add_Ds(std::move(D_output), D_result, fac);
			}
		}
	}

	~Output()
	{
		GPU_Backend::free(this->d_data);
		GPU_Backend::free(this->d_array);
	}

  private:
	static void download_data_impl(
		Tensor<Tdata_CPU> &D_output,
		const std::vector<Tdata_GPU> &h_data,
		const Pack &pack,
		std::true_type)
	{
		std::memcpy(
			D_output.ptr(),
			h_data.data()+pack.pos,
			pack.shape.get_shape_all()*sizeof(Tdata_CPU));
	}

	static void download_data_impl(
		Tensor<Tdata_CPU> &D_output,
		const std::vector<Tdata_GPU> &h_data,
		const Pack &pack,
		std::false_type)
	{
		const std::size_t size = pack.shape.get_shape_all();
		const auto ptr_src = h_data.begin()+pack.pos;
		const auto ptr_dest = D_output.ptr();
		for(std::size_t i=0; i<size; ++i)
			ptr_dest[i] = GPU_Backend::data_to_CPU(ptr_src[i]);
	}

  public:
	std::size_t totalSize = 0;                  // 总的数据数量
	Tdata_GPU *d_data = nullptr;					// 存储数据（GPU）
	std::vector<Pack> h_array;					// 记录每个batch的Pack
	Tdata_GPU **d_array = nullptr;					// 记录每个batch的 d_data 指针（GPU）
	std::map<TA, std::map<TAC, Pack>> ptrList;	// 记录每个原子对的Pack
};

}

}

#endif
