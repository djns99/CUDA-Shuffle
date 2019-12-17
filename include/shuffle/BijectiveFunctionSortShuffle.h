#pragma once
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "shuffle/Shuffle.h"

template <class BijectiveFunction, class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class BijectiveFunctionSortShuffle : public Shuffle<ContainerType, RandomGenerator>
{
private:
    static uint64_t roundUpPower2( uint64_t a )
    {
        if( a & ( a - 1 ) )
        {
            uint64_t i;
            for( i = 0; a > 1; i++ )
            {
                a >>= 1ull;
            }
            return 1ull << ( i + 1ull );
        }
        return a;
    }

public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        if( &in_container != &out_container )
        {
            // Copy if we are not doing an inplace operation
            thrust::copy( thrust::device, in_container.begin(), in_container.begin() + num,
                          out_container.begin() );
        }

        RandomGenerator random_function( seed );
        BijectiveFunction mapping_function;
        mapping_function.init( num, random_function );

        thrust::device_vector<uint64_t> keys( num );

        // Initialise key vector with indexes
        thrust::sequence( thrust::device, keys.begin(), keys.end() );
        // Inplace transform
        thrust::transform( thrust::device, keys.begin(), keys.end(), keys.begin(),
                           [mapping_function] __host__ __device__( uint64_t val ) -> uint64_t {
                               return mapping_function( val );
                           } );
        // Sort by keys
        thrust::sort_by_key( thrust::device, keys.begin(), keys.end(), out_container.begin() );
    }
};