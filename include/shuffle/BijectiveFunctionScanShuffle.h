#pragma once
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"

template <class BijectiveFunction, class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class BijectiveFunctionScanShuffle : public Shuffle<ContainerType, RandomGenerator>
{
public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        assert( &in_container != &out_container );

        RandomGenerator random_function( seed );
        BijectiveFunction mapping_function;
        mapping_function.init( num, random_function );
        uint64_t capacity = mapping_function.getMappingRange();

        thrust::counting_iterator<uint64_t> indexes( 0 );
        thrust::device_vector<uint64_t> mappings( capacity );
        auto mapping_it = mappings.begin();
        thrust::transform( indexes, indexes + capacity, mapping_it,
                           [mapping_function] __host__ __device__( uint64_t val ) -> uint64_t {
                               // Call mapping functions
                               return mapping_function( val );
                           } );

        // Transform iterator for low memory footprint - worse performance
        // auto mapping_it =
        //     thrust::make_transform_iterator( indexes, [mapping_function] __host__ __device__( uint64_t val ) -> uint64_t {
        //         // Call mapping functions
        //         return mapping_function( val );
        //     } );

        auto is_valid_it =
            thrust::make_transform_iterator( mapping_it, [num] __host__ __device__( uint64_t val ) -> bool {
                // Check if mapping is valid
                return val < num;
            } );

        thrust::device_vector<uint64_t> scan_buf( capacity );
        thrust::exclusive_scan( is_valid_it, is_valid_it + capacity, scan_buf.begin() );
        auto out_permutation_iterator =
            thrust::make_permutation_iterator( out_container.begin(), scan_buf.begin() );

        thrust::gather_if( mapping_it, mapping_it + capacity, is_valid_it, in_container.begin(),
                           out_permutation_iterator );
    }

    bool supportsInPlace() override
    {
        return false;
    }
};