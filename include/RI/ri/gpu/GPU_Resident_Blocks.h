#ifndef RI_GPU_RESIDENT_BLOCKS_H
#define RI_GPU_RESIDENT_BLOCKS_H

#ifdef __GPU_RI

#include "../../global/gpu/GPU_Backend.h"
#include "../../global/Tensor.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace RI
{
namespace GPU_Data
{

// Owns geometry-fixed input blocks on one GPU. The caller invalidates a name
// whenever its C/V tensors are replaced; D tensors are never registered here.
template<typename TA, typename TAC, typename Tdata_CPU>
class Resident_Blocks
{
  public:
    using Tdata_GPU = GPU_Backend::Type_to_GPU<Tdata_CPU>;
    using Matrix_Map = std::map<TA, std::map<TAC, Tensor<Tdata_CPU>>>;
    using Block_Map = std::map<TA, std::map<TAC, Tdata_GPU*>>;

    explicit Resident_Blocks(const MPI_Comm &communicator)
    {
        this->context.reset(new GPU_Backend::Context(communicator));
#ifdef __DDLA_RI
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if(deviceMemGetInfo(&free_bytes, &total_bytes) == ddla::deviceSuccess)
        {
            const std::size_t cap = std::size_t(4) * 1024 * 1024 * 1024;
            this->budget_bytes = std::min(free_bytes / 3, cap);
        }
#endif
    }

    GPU_Backend::Queue& queue() { return this->context->queue(); }

    // If caches hits, return the key-address table;
    // else, allocate device memory, build the table and upload the data.
    // Return null when the budget cannot hold the whole named tensor pack.
    // Cached pointers remain valid until invalidate() or destruction.
    const Block_Map* ensure(const std::string &name, int variant, const Matrix_Map &source)
    {
        const Key key(name, variant);
        const auto found = this->packs.find(key);
        if(found != this->packs.end()) return &found->second->blocks;

        std::size_t count = 0;
        for(const auto &row : source)
            for(const auto &block : row.second)
                count += block.second.shape.get_shape_all();
        const std::size_t bytes = count * sizeof(Tdata_GPU);
        if(bytes > this->budget_bytes - this->used_bytes) return nullptr;

        std::unique_ptr<Pack> pack(new Pack);
        pack->bytes = bytes;
        std::vector<Tdata_GPU> host(count);
        GPU_Backend::allocate(&pack->data, count);
        std::size_t offset = 0;
        for(const auto &row : source)
        {
            for(const auto &block : row.second)
            {
                const Tensor<Tdata_CPU> &tensor = block.second;
                const std::size_t block_size = tensor.shape.get_shape_all();
                pack->blocks[row.first][block.first] = block_size ? pack->data + offset : nullptr;
                if(block_size) copy_tensor(host.data() + offset, tensor, Same_data_layout());
                offset += block_size;
            }
        }
        GPU_Backend::upload(count, host.data(), pack->data, this->context->queue());
        GPU_Backend::sync(this->context->queue());
        this->used_bytes += bytes;
        const Block_Map *blocks = &pack->blocks;
        this->packs.emplace(key, std::move(pack));
        return blocks;
    }

    void invalidate(const std::string &name)
    {
        for(auto it = this->packs.begin(); it != this->packs.end();)
        {
            if(it->first.first == name)
            {
                this->used_bytes -= it->second->bytes;
                it = this->packs.erase(it);
            }
            else ++it;
        }
    }

  private:
    using Key = std::pair<std::string, int>;
    using Same_data_layout = std::integral_constant<bool,
        sizeof(Tdata_CPU) == sizeof(Tdata_GPU) && alignof(Tdata_CPU) == alignof(Tdata_GPU)>;

    struct Pack
    {
        Tdata_GPU *data = nullptr;
        Block_Map blocks;
        std::size_t bytes = 0;
        ~Pack() { GPU_Backend::free(data); }
    };

    static void copy_tensor(Tdata_GPU *dest, const Tensor<Tdata_CPU> &tensor, std::true_type)
    {
        std::copy(tensor.ptr(), tensor.ptr() + tensor.shape.get_shape_all(), dest);
    }

    static void copy_tensor(Tdata_GPU *dest, const Tensor<Tdata_CPU> &tensor, std::false_type)
    {
        const Tdata_CPU *source = tensor.ptr();
        for(std::size_t i = 0; i < tensor.shape.get_shape_all(); ++i)
            dest[i] = GPU_Backend::data_to_GPU(source[i]);
    }

    std::unique_ptr<GPU_Backend::Context> context;
    std::map<Key, std::unique_ptr<Pack>> packs;
    std::size_t budget_bytes = 0;
    std::size_t used_bytes = 0;
};

}
}

#endif
#endif
