// =================================
//  Author: Laiyuan Yang, Peize Lin
//  date: 2026.01.01
// =================================

#pragma once
#include <omp.h>
#include <map>
#include <memory_resource>
#include <vector>
#include "../../global/Shape_Vector.h"

#ifdef __GPU_RI

namespace RI
{

namespace GPU_Data
{

// Release host upload sources only after the caller's existing queue sync.
// This adds no synchronization and leaves cached immutable layouts untouched.
template<typename... Workspaces>
void release_completed_uploads(Workspaces&... workspaces)
{ (workspaces.release_completed_uploads(),...); }

struct Pack
{
	bool exist = false;
	std::size_t pos = -1;
	Shape_Vector shape;
	int thread_num = 0;
};
// Related operands and dimensions append in the same worker-local order.
// Flat arrays own values; caches never retain pointers into these temporary chunks.
template<typename T>
class Thread_Batch
{
    struct alignas(64) Chunk { std::vector<T> values; };
    std::vector<Chunk> chunks{static_cast<std::size_t>(omp_get_max_threads())};
public:
    void push_back(const T &value) { chunks[omp_get_thread_num()].values.push_back(value); }
    std::size_t size() const { std::size_t n=0; for(const auto &c:chunks) n+=c.values.size(); return n; }
    template<typename F> void for_each(F f) const { for(const auto &c:chunks) for(const auto &v:c.values) f(v); }
    void release() { for(auto &c:chunks) std::vector<T>().swap(c.values); }
};

// Register unique pairs privately, then resolve one shared offset per block.
template<typename TA, typename TAC, typename Record=Pack>
class Pair_Packs
{
    // Each worker owns an arena; map nodes are destroyed before bulk release.
    using Local_Map=std::pmr::map<TA,std::pmr::map<TAC,Record>>;
    struct alignas(64) Shard
    {
        std::pmr::monotonic_buffer_resource storage;
        Local_Map values{&storage};
    };
    std::vector<Shard> shards{static_cast<std::size_t>(omp_get_max_threads())};
protected:
    auto& local_packs() { return shards[omp_get_thread_num()].values; }
    template<typename F> void merge_packs(F first_insert)
    {
        for(auto &shard:shards) for(auto &row:shard.values) for(auto &block:row.second)
        {
            Record &local=block.second;
            Pack &shared=ptrList[row.first][block.first];
            if(!shared.exist)
            {
                shared=local; shared.pos=totalSize;
                totalSize+=shared.shape.get_shape_all();
                first_insert(local,shared);
            }
            static_cast<Pack&>(local)=shared;
        }
    }
    void merge_packs() { merge_packs([](const Record&,Pack&){}); }
    void release_local_packs() { for(auto &shard:shards) { shard.values.clear(); shard.storage.release(); } }
public:
    using Pack_Map = std::map<TA,std::map<TAC,Pack>>;
    // Cache owns immutable pair offsets; work objects borrow them only for this call.
    const Pack_Map& pairs() const { return cached_pairs ? *cached_pairs : ptrList; }
    void bind_pairs(const Pack_Map &layout) { cached_pairs = &layout; }
    std::size_t totalSize=0;
    Pack_Map ptrList;
private:
    const Pack_Map *cached_pairs = nullptr;
};


}

}

#endif
