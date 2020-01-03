#pragma once
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/generate.h>
#include <thrust/sort.h>

#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"
#include <algorithm>

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
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

        const uint64_t internal_seed = RandomGenerator( seed )();
        thrust::counting_iterator<uint64_t> offsets;
        thrust::device_vector<uint64_t> d_keys( num );
        thrust::transform( thrust::device, offsets, offsets + num, d_keys.begin(),
                           [internal_seed] __device__( uint64_t idx ) {
                               // Lehmar
                               constexpr uint64_t combiner_constant = 0xda942042e4dd58b5;
                               uint64_t tmp = __umul64hi( internal_seed, combiner_constant ) + idx * combiner_constant;
                               // Wyhash
                               tmp += 0x60bee2bee120fc15;
                               constexpr uint64_t mul_constant1 = 0xa3b195354a39b70d;
                               tmp = __umul64hi( tmp, mul_constant1 ) ^ ( tmp * mul_constant1 );
                               constexpr uint64_t mul_constant2 = 0x1b03738712fad5c9;
                               tmp = __umul64hi( tmp, mul_constant2 ) ^ ( tmp * mul_constant2 );
                               return tmp;
                           } );

        // Sort by keys
        thrust::sort_by_key( thrust::device, d_keys.begin(), d_keys.end(), out_container.begin() );
    }
};