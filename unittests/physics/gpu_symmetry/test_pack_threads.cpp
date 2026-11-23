#ifdef NDEBUG
#undef NDEBUG
#endif
#include <RI/ri/LRI.h>
#include <RI/ri/gpu/Dim.h>
#include <RI/ri/gpu/GPU_Data_Input.h>
#include <RI/ri/gpu/GPU_Data_Mul.h>
#include <RI/ri/gpu/GPU_Data_Output.h>
#include <RI/ri/gpu/GPU_Data_Tmp.h>

#include <cassert>
#include <iostream>

using Cell = std::array<int, 1>;
using Key = std::pair<int, Cell>;

template <typename T>
void check(const char* scalar)
{
    RI::GPU_Backend::Context context(MPI_COMM_WORLD);
    auto queue = context.queue();
    constexpr int count = 257;
    constexpr int keys = 7;
    std::vector<RI::Tensor<T>> tensors;
    for (int k = 0; k < keys; ++k)
    {
        tensors.emplace_back(RI::Shape_Vector{std::size_t(k + 1), 2});
        for (std::size_t j = 0; j < tensors.back().data->size(); ++j)
            (*tensors.back().data)[j] = T(100 * k + j);
    }
    RI::GPU_Data::Input<int, Key, T> input;
    RI::GPU_Data::Mul<int, Key, T> mul;
    RI::GPU_Data::Output<int, Key, T> output;
    RI::GPU_Data::Tmp<int, Key, T> tmp;
    RI::Dim_mnk dim;
    std::vector<std::vector<int>> order(omp_get_max_threads());
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < count; ++i)
    {
        const int k = i % keys;
        order[omp_get_thread_num()].push_back(k);
        input.insert(0, {k, {0}}, tensors[k]);
        input.insert_data(tensors[k]);
        mul.insert_1st(0, {k, {0}}, tensors[k].shape);
        output.insert(0, {k, {0}}, tensors[k].shape);
        tmp.insert({std::size_t(k + 1)});
        dim.input(k + 1, 2, 3);
    }
    std::vector<int> expected;
    for (const auto& worker : order) expected.insert(expected.end(), worker.begin(), worker.end());
    input.upload(queue);
    mul.upload_1st(queue);
    output.upload(queue);
    const auto segments = tmp.upload(8 * sizeof(T), queue);
    dim.upload(queue);
    assert(input.ptrList.at(0).size() == keys);
    assert(mul.ptrList.at(0).size() == keys);
    assert(output.ptrList.at(0).size() == keys);
    assert(input.batch_block_ids.size() == count && mul.first_batch_offsets.size() == count);
    assert(output.batch_offsets.size() == count && tmp.h_array.size() == count);

    std::vector<T*> in_ptrs(count), mul_ptrs(count), out_ptrs(count), tmp_ptrs(count);
    std::vector<int> dimensions(count);
    std::vector<T> values(input.totalSize);
    RI::GPU_Backend::download(count, input.d_array, in_ptrs.data(), queue);
    RI::GPU_Backend::download(count, mul.d_array_1, mul_ptrs.data(), queue);
    RI::GPU_Backend::download(count, output.d_array, out_ptrs.data(), queue);
    RI::GPU_Backend::download(count, tmp.d_array, tmp_ptrs.data(), queue);
    RI::GPU_Backend::download(count, dim.m.data(), dimensions.data(), queue);
    RI::GPU_Backend::download(values.size(), input.d_data, values.data(), queue);
    RI::GPU_Backend::sync(queue);
    // Cached contribution sequences own offsets, not pointers into temporary shards.
    const auto cached_mul_pairs=std::move(mul.ptrList);
    const auto cached_mul_offsets=std::move(mul.first_batch_offsets);
    mul.bind_first(cached_mul_pairs,cached_mul_offsets);
    assert(mul.first_offsets().data()==cached_mul_offsets.data());
    const auto cached_output_pairs=std::move(output.ptrList);
    const auto cached_output_offsets=std::move(output.batch_offsets);
    output.bind_layout(cached_output_pairs,cached_output_offsets);
    assert(output.offsets().data()==cached_output_offsets.data());
    const auto input_layout = input.take_layout();
    input.bind_layout(input_layout);
    typename RI::GPU_Data::Input<int, Key, T>::Matrix_Map changed_source;
    for(int k=0;k<keys;++k)
    {
        RI::Tensor<T> changed(tensors[k].shape);
        for(std::size_t j=0;j<changed.data->size();++j) (*changed.data)[j]=T(200*k+j);
        changed_source[0][{k,{0}}]=std::move(changed);
    }
    RI::GPU_Data::Input<int, Key, T> cached_input;
    cached_input.load_layout(input_layout,changed_source);
    assert(&cached_input.pairs()==&input_layout.ptr_list);
    assert(cached_input.block_ids().data()==input_layout.block_ids.data());
    assert(cached_input.slot_sizes().data()==input_layout.sizes.data());
    cached_input.upload(queue);
    std::vector<T> changed_values(values.size());
    RI::GPU_Backend::download(changed_values.size(),cached_input.d_data,changed_values.data(),queue);
    RI::GPU_Backend::sync(queue);
    std::vector<std::size_t> input_slot_begin(input_layout.sizes.size()+1,0);
    for(std::size_t slot=0;slot<input_layout.sizes.size();++slot)
        input_slot_begin[slot+1]=input_slot_begin[slot]+input_layout.sizes[slot];
    std::vector<RI::GPU_Backend::Int> expected_segments;
    std::size_t used = 0;
    int batch = 0;
    for (int i = 0; i < count; ++i)
    {
        const int k = expected[i];
        const Key key{k, {0}};
        const auto& input_pack = input.pairs().at(0).at(key);
        const auto offset = input_slot_begin.at(input_pack.thread_num) + input_pack.pos;
        assert(in_ptrs[i] == input.d_data + offset);
        assert(mul_ptrs[i] == mul.d_data + mul.pairs().at(0).at(key).pos);
        assert(out_ptrs[i] == output.d_data + output.pairs().at(0).at(key).pos);
        assert(dimensions[i] == k + 1);
        for (std::size_t j = 0; j < tensors[k].data->size(); ++j)
        {
            assert(values[offset + j] == (*tensors[k].data)[j]);
            assert(changed_values[offset+j]==T(200*k+j));
        }
        if (used + k + 1 > 8)
        {
            expected_segments.push_back(batch);
            batch = 0;
            used = 0;
        }
        assert(tmp_ptrs[i] == tmp.d_data + used);
        used += k + 1;
        ++batch;
    }
    expected_segments.push_back(batch);
    assert(segments == expected_segments);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) mul.insert_2nd(mul.find_2nd(0, {i % keys, {0}}));
    mul.upload_2nd(queue);
    assert(mul.second_batch_offsets.size() == count);
    std::map<int, std::map<Key, RI::Tensor<T>>> result;
    output.download(result, queue, T(1));
    assert(result.at(0).size() == keys);
    for (const auto& block : result.at(0))
        for (const auto value : *block.second.data) assert(value == T(0));
    // The same cached IDs must bind to the current resident addresses, not the
    // addresses from the call that built the layout. Neither resident input owns values.
    for(auto *source_input:{&input,&cached_input})
    {
        typename RI::GPU_Data::Input<int,Key,T>::Block_Map resident_addresses;
        for(const auto &row:input_layout.ptr_list) for(const auto &entry:row.second)
        {
            const auto &pack=entry.second;
            resident_addresses[row.first][entry.first]=source_input->d_data+
                input_slot_begin[pack.thread_num]+pack.pos;
        }
        RI::GPU_Data::Input<int,Key,T> resident_input;
        resident_input.bind_resident_blocks(&resident_addresses);
        resident_input.load_layout(input_layout,changed_source);
        resident_input.upload(queue);
        std::vector<T*> resident_pointers(count);
        RI::GPU_Backend::download(count,resident_input.d_array,resident_pointers.data(),queue);
        RI::GPU_Backend::sync(queue);
        assert(resident_input.d_data==nullptr);
        assert(resident_input.blocks().size()==keys);
        for(int i=0;i<count;++i)
        {
            const auto &pack=input_layout.ptr_list.at(0).at({expected[i],{0}});
            assert(resident_pointers[i]==source_input->d_data+
                input_slot_begin[pack.thread_num]+pack.pos);
        }
    }
    // Releasing completed host transfers must not invalidate cache aliases,
    // device inputs, Mul pair lookups, or Output block reconstruction.
    RI::GPU_Data::release_completed_uploads(input,cached_input,mul,output,tmp,dim);
    assert(!input.host_values && input.upload_pointers.empty());
    assert(cached_input.block_ids().data()==input_layout.block_ids.data());
    assert(mul.first_offsets().data()==cached_mul_offsets.data());
    assert(output.offsets().data()==cached_output_offsets.data());
    assert(mul.find_2nd(0,{0,{0}}).exist);
    std::map<int,std::map<Key,RI::Tensor<T>>> after_release;
    output.download(after_release,queue,T(1));
    assert(after_release.at(0).size()==keys);
    for(const auto &block:after_release.at(0))
        for(const auto value:*block.second.data) assert(value==T(0));
    RI::GPU_Data::Tmp<int, Key, T> empty;
    assert(empty.upload(8 * sizeof(T), queue).empty());
    std::cout << "PACK_THREADS_PASS " << scalar << " threads=" << omp_get_max_threads()
              << " entries=" << count << " segments=" << segments.size() << '\n';
}

// Device dimensions survive the allocating call/context owner and are reused
// from a new queue without changing their addresses or values.
void check_dimension_lifetime()
{
    auto first_context=std::make_shared<RI::GPU_Backend::Context>(MPI_COMM_WORLD);
    std::weak_ptr<RI::GPU_Backend::Context> owner=first_context;
    RI::Dim_mnk::Layout layout;
    int *first_m=nullptr;
    {
        RI::Dim_mnk initial;
        initial.input(3,4,5); initial.input(7,8,9);
        layout=initial.save_layout();
        initial.upload(first_context->queue(),first_context);
        first_m=initial.m.data();
        assert(layout.m->shapes==std::vector<int>({3,7}));
    }
    first_context.reset();
    assert(!owner.expired());
    {
        auto next_context=std::make_shared<RI::GPU_Backend::Context>(MPI_COMM_WORLD);
        RI::Dim_mnk restored;
        restored.load_layout(layout);
        restored.upload(next_context->queue(),next_context);
        assert(restored.m.data()==first_m);
        std::vector<int> m(2),n(2),k(2);
        auto q=next_context->queue();
        RI::GPU_Backend::download(2,restored.m.data(),m.data(),q);
        RI::GPU_Backend::download(2,restored.n.data(),n.data(),q);
        RI::GPU_Backend::download(2,restored.k.data(),k.data(),q);
        RI::GPU_Backend::sync(q);
        assert(m==std::vector<int>({3,7}));
        assert(n==std::vector<int>({4,8}));
        assert(k==std::vector<int>({5,9}));
        layout={};
        assert(!owner.expired());
    }
    assert(owner.expired());
    std::cout<<"CACHED_DIMENSIONS_PASS\n";
}

int main(int argc, char** argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    assert(provided >= MPI_THREAD_MULTIPLE);
    assert(RI::MPI_Wrapper::mpi_get_size(MPI_COMM_WORLD) == 1);
    check_dimension_lifetime();
    check<float>("float");
    check<double>("double");
    check<std::complex<float>>("complex_float");
    check<std::complex<double>>("complex_double");
    MPI_Finalize();
}
