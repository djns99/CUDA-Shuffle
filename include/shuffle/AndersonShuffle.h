#pragma once
#include "DefaultRandomGenerator.h"
#include "ThrustInclude.h"
#include "shuffle/Shuffle.h"
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

template <class Data>
__global__ void andersonShuffle( Data* data, uint32_t* guards, GPURandomGenerator* gens, uint64_t num, uint64_t original_gens, uint64_t seed )
{
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if( tid >= num - 1 )
        return;

    if( tid >= original_gens )
        new( gens + tid ) GPURandomGenerator( seed, tid );

    uint64_t swap_index = gens[tid]() % ( num - tid );

    if( swap_index == 0 )
        return;

    swap_index += tid;

    const uint64_t my_word = tid / 32;
    const uint64_t my_off = tid % 32;

    // Sketchy spin lock
    const uint32_t my_shift = 1ull << my_off;
    while( atomicOr( guards + my_word, my_shift ) & my_shift )
        ;

    const uint64_t swap_word = swap_index / 32;
    const uint64_t swap_off = swap_index % 32;

    const uint32_t swap_shift = 1ull << swap_off;
    while( atomicOr( guards + swap_word, swap_shift ) & swap_shift )
        ;

    auto tmp = std::move( data[tid] );
    data[tid] = std::move( data[swap_index] );
    data[swap_index] = std::move( tmp );

    // Release said spin lock
    atomicAnd( guards + swap_word, ~swap_shift );
    atomicAnd( guards + my_word, ~my_shift );
}

template <class ContainerType = thrust::device_vector<uint64_t>>
class AndersonShuffle : public Shuffle<ContainerType, GPURandomGenerator>
{
private:
    thrust::device_vector<uint32_t> temp_storage;
    thrust::device_vector<GPURandomGenerator> gen_cache;

public:
    virtual void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        const uint64_t old_size = temp_storage.size();
        if( gen_cache.size() < num )
        {
            temp_storage.resize( ( num + 31 ) / 32 );
            gen_cache.resize( num );
        }

        if( &in_container != &out_container )
        {
            // Copy if we are not doing an inplace operation
            thrust::copy( in_container.begin(), in_container.begin() + num, out_container.begin() );
        }


        andersonShuffle<<<( num + 255 ) / 256, 256>>>( out_container.data().get(),
                                                       temp_storage.data().get(),
                                                       gen_cache.data().get(), num, old_size, seed );
        cudaDeviceSynchronize();
    }

    bool supportsInPlace() const override
    {
        return true;
    }

    bool isDeterministic() const override
    {
        return false;
    }
};