#pragma once
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"

template <class BijectiveFunction, class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class BijectiveFunctionShuffle : public Shuffle<ContainerType, RandomGenerator>
{
public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        assert( &in_container != &out_container );

        RandomGenerator random_function( seed );
        BijectiveFunction mapping_function;
        mapping_function.init( num, random_function );

        thrust::counting_iterator<uint64_t> indexes( 0 );

        auto transform_it_begin =
            thrust::make_transform_iterator( indexes, [mapping_function] __host__ __device__( uint64_t val ) -> uint64_t {
                // Call mapping functions
                return mapping_function( val );
            } );
        auto transform_it_end = transform_it_begin + num;

        thrust::copy( thrust::make_permutation_iterator( in_container.begin(), transform_it_begin ),
                      thrust::make_permutation_iterator( in_container.begin(), transform_it_end ),
                      out_container.begin() );
        // thrust::copy( in_container.begin(), in_container.end(),
        //               thrust::make_permutation_iterator( out_container.begin(),
        //               transform_it_begin ) );
    }

    bool supportsInPlace() const override
    {
        return false;
    }
};