// ===================
//  Author: Peize Lin
//  date: 2026.02.18
// ===================

#pragma once

#ifdef __GPU_RI

#include "../../global/gpu/GPU_Backend.h"
#include <vector>
#include <memory>
#include <utility>
#include <stdexcept>
#include "GPU_Data_Pack.h"

namespace RI
{

class Dim
{
    GPU_Data::Thread_Batch<int> pending;
  public:
    // Immutable GEMM dimensions and their device allocation share cache lifetime.
    // Keep the allocating context alive; future calls may use another queue on this GPU.
    class Layout
    {
      public:
        explicit Layout(std::vector<int> &&values) : shapes(std::move(values)) {}
        Layout(const Layout&) = delete;
        Layout& operator=(const Layout&) = delete;
        const std::vector<int> shapes;

        void upload_once(const std::shared_ptr<GPU_Backend::Context> &context)
        {
            if(device_values) return;
            if(!context)
                throw std::logic_error("cached GEMM dimensions need their allocating context");
            const auto queue = context->queue();
            int *values = nullptr;
            try
            {
                // Preserve the extra element required by supported variable-batch backends.
                GPU_Backend::allocate(&values, shapes.size()+1, queue);
                GPU_Backend::upload(shapes.size(), shapes.data(), values, queue);
                // Publish complete arrays before a later call uses another queue.
                GPU_Backend::sync(queue);
            }
            catch (...)
            {
                GPU_Backend::free(values, queue);
                throw;
            }
            owner = context;
            device_values = values;
        }
        int* data() const { return device_values; }
        ~Layout()
        {
            if(device_values) GPU_Backend::free(device_values, owner->queue());
        }
      private:
        std::shared_ptr<GPU_Backend::Context> owner;
        int *device_values = nullptr;
    };

    void input(int shape) { pending.push_back(shape); }
    void input(int shape1, int shape2) { pending.push_back(shape1*shape2); }
    void prepare_layout()
    {
        pending.for_each([&](int shape){shapes.push_back(shape);}); pending.release();
    }
    std::shared_ptr<Layout> save_layout()
    {
        if(cached_layout) return cached_layout;
        if(d_m) throw std::logic_error("freeze dimensions before their first upload");
        prepare_layout();
        cached_layout = std::make_shared<Layout>(std::move(shapes));
        return cached_layout;
    }
    void load_layout(const std::shared_ptr<Layout> &layout)
    {
        if(!cached_layout && d_m)
            throw std::logic_error("bind dimension layouts before their first upload");
        cached_layout = layout;
        d_m = nullptr;
    }
    void upload(GPU_Backend::Queue queue,
                const std::shared_ptr<GPU_Backend::Context> &owner = {})
    {
        queue_ = queue;
        if(cached_layout)
        {
            cached_layout->upload_once(owner);
            d_m = cached_layout->data();
            return;
        }
        prepare_layout();
        GPU_Backend::allocate(&d_m, shapes.size()+1, queue);
        GPU_Backend::upload(shapes.size(), shapes.data(), d_m, queue);
    }
    void release_completed_uploads()
    { if(!cached_layout) std::vector<int>().swap(shapes); }
    ~Dim() { if(!cached_layout) GPU_Backend::free(d_m, queue_); }
    int* data() const { return d_m; }

    GPU_Backend::Queue queue_ = nullptr;
    std::vector<int> shapes;
    int* d_m = nullptr;
  private:
    std::shared_ptr<Layout> cached_layout;
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

    void release_completed_uploads()
    { m.release_completed_uploads(); n.release_completed_uploads(); k.release_completed_uploads(); }
    struct Layout
    {
        std::shared_ptr<Dim::Layout> m, n, k;
    };
    Layout save_layout() { return {m.save_layout(),n.save_layout(),k.save_layout()}; }
    void load_layout(const Layout &layout)
    { m.load_layout(layout.m); n.load_layout(layout.n); k.load_layout(layout.k); }
    void upload(GPU_Backend::Queue queue,
                const std::shared_ptr<GPU_Backend::Context> &owner = {})
    {
        m.upload(queue, owner); n.upload(queue, owner); k.upload(queue, owner);
    }
};

}

#endif
