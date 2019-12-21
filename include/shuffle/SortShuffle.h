#pragma once
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/generate.h>
#include <thrust/sort.h>

#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"
#include <algorithm>

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = GPURandomGenerator>
class SortShuffle : public Shuffle<ContainerType, RandomGenerator>
{
public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        if( &in_container != &out_container )
        {
            // Copy if we are not doing an inplace operation
            thrust::copy( in_container.begin(), in_container.begin() + num, out_container.begin() );
        }

        thrust::counting_iterator<uint64_t> offsets;
        thrust::device_vector<uint64_t> d_keys( num );
        thrust::transform( thrust::device, offsets, offsets + num, d_keys.begin(),
                           [seed] __device__( uint64_t idx ) {
                               RandomGenerator gen( seed, idx );
                               return gen();
                           } );

        // Sort by keys
        thrust::sort_by_key( d_keys.begin(), d_keys.end(), out_container.begin() );
    }
};