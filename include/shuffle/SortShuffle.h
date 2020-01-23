#pragma once
#include <thrust/device_vector.h>

#include "WyHash.h"
#include "shuffle/Shuffle.h"
#include <cub/device/device_radix_sort.cuh>
#include <thrust/transform.h>

template <class BijectiveFunction, class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
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
                           [seed] __device__( uint64_t val ) -> uint64_t {
                               return WyHash::wyhash64_v3_pair( val, seed );
                           } );

        // Determine temporary device storage requirements
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs( NULL, temp_storage_bytes, key_in.data().get(),
                                         key_out.data().get(), in_container.data().get(),
                                         out_container.data().get(), num );

        if( temp_storage_bytes > temp_storage.size() )
        {
            temp_storage.resize( temp_storage_bytes );
        }

        cub::DeviceRadixSort::SortPairs( temp_storage.data().get(), temp_storage_bytes,
                                         key_in.data().get(), key_out.data().get(),
                                         in_container.data().get(), out_container.data().get(), num );
    }

    bool supportsInPlace() const override
    {
        return false;
    }
};