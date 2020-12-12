#pragma once
#include "ThrustInclude.h"
#include "WyHash.h"
#include "shuffle/Shuffle.h"
#include <cub/device/device_radix_sort.cuh>


using GPURand = thrust::random::taus88;

template <class ValueType>
__global__ void fisherYatesIdenticalKey( uint64_t* keys, ValueType* values, uint64_t num, uint64_t key1, uint64_t key2 )
{
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if( tid >= num - 1 )
        return;
    auto me = keys[tid];
    if( me != keys[tid + 1] )
        return;
    if( tid != 0 && keys[tid - 1] == me )
        return;
    uint64_t key[2] = { key1, key2 };
    GPURand rng( WyHash::wyhash64_v4_key2( key, tid ) );
    uint64_t i;
    for( i = tid; i < num && keys[i] == me; i++ )
        ;

    for( ; tid < i - 1; tid++ )
    {
        uint64_t next = rng() % ( i - tid );
        auto copy = values[tid];
        values[tid] = values[tid + next];
        values[tid + next] = copy;
    }
}

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class SortShuffle : public Shuffle<ContainerType, RandomGenerator>
{
private:
    thrust::device_vector<uint8_t> temp_storage;
    thrust::device_vector<uint64_t> key_in;
    thrust::device_vector<uint64_t> key_out;

public:
    SortShuffle()
    {
        temp_storage.resize( 1 << 16 );
        key_in.resize( 1 << 16 );
        key_out.resize( 1 << 16 );
    }

    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        RandomGenerator g( seed );
        uint64_t key1_1 = g();
        uint64_t key1_2 = g();
        uint64_t key2_1 = g();
        uint64_t key2_2 = g();
        if( num > key_in.size() )
        {
            key_in.resize( num );
            key_out.resize( num );
        }

        // Initialise key vector with indexes
        thrust::counting_iterator<uint64_t> counting_begin( 0 );
        auto counting_end = counting_begin + num;
        // Inplace transform
        thrust::transform( thrust::device, counting_begin, counting_end, key_in.begin(),
                           [key1_1, key1_2, num] __device__( uint64_t val ) -> uint64_t {
                               uint64_t key[2] = { key1_1, key1_2 };
                               return WyHash::wyhash64_v4_key2( key, val ) % num;
                           } );

        // Determine temporary device storage requirements
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs( NULL, temp_storage_bytes, key_in.data().get(),
                                         key_out.data().get(), in_container.data().get(),
                                         out_container.data().get(), num, 0, 64 - __builtin_clzll( num ) );

        if( temp_storage_bytes > temp_storage.size() )
        {
            temp_storage.resize( temp_storage_bytes );
        }

        cub::DeviceRadixSort::SortPairs( temp_storage.data().get(), temp_storage_bytes,
                                         key_in.data().get(), key_out.data().get(),
                                         in_container.data().get(), out_container.data().get(), num,
                                         0, 64 - __builtin_clzll( num ) );

        fisherYatesIdenticalKey<uint64_t>
            <<<( num + 255 ) / 256, 256, 0, 0>>>( key_out.data().get(), out_container.data().get(),
                                                  num, key2_1, key2_2 );
    }

    bool supportsInPlace() const override
    {
        return false;
    }
};
