// ===================
//  Author: Peize Lin
//  date: 2026.02.18
// ===================

#pragma once

#ifdef __GPU_RI

#include "../../global/gpu/GPU_Backend.h"
#include <vector>

namespace RI
{

class Dim
{
  public:
	void input(const int shape)
	{
		this->shapes.push_back(shape);
	}
	void input(const int shape1, const int shape2)
	{
		this->shapes.push_back(shape1 * shape2);
	}

	void upload(GPU_Backend::Queue queue)
	{
		this->queue_ = queue;
		GPU_Backend::allocate(&this->d_m, this->shapes.size() + 1, queue);
		GPU_Backend::upload(
			this->shapes.size(), this->shapes.data(), this->d_m, queue);
	}

	~Dim()
	{
		GPU_Backend::free(this->d_m, this->queue_);
	}

	int* data() const
	{
		return this->d_m;
	}

	std::vector<int> shapes;
	int* d_m = nullptr;

  private:
	// The context must outlive the buffers allocated on its queue.
	GPU_Backend::Queue queue_ = nullptr;
};


class Dim_mnk
{
  public:
	Dim m;
	Dim n;
	Dim k;

	void input(const int shape_m, const int shape_n, const int shape_k)
	{
		m.input(shape_m);
		n.input(shape_n);
		k.input(shape_k);
	}

	void input(const std::pair<int,int> shape_m, const int shape_n, const int shape_k)
	{
		m.input(std::get<0>(shape_m), std::get<1>(shape_m));
		n.input(shape_n);
		k.input(shape_k);
	}

	void input(const int shape_m, const std::pair<int,int> shape_n, const int shape_k)
	{
		m.input(shape_m);
		n.input(std::get<0>(shape_n), std::get<1>(shape_n));
		k.input(shape_k);
	}

	void input(const int shape_m, const int shape_n, const std::pair<int,int> shape_k)
	{
		m.input(shape_m);
		n.input(shape_n);
		k.input(std::get<0>(shape_k), std::get<1>(shape_k));
	}

	void upload(GPU_Backend::Queue queue)
	{
		m.upload(queue);
		n.upload(queue);
		k.upload(queue);
	}
};

}

#endif
