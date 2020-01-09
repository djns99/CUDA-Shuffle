#pragma once
#include <thrust/device_vector.h>

#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"
#include <cub/cub.cuh>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>


template <class BijectiveFunction, class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class BijectiveFunctionScanShuffle : public Shuffle<ContainerType, RandomGenerator>
{
    thrust::device_vector<uint8_t> temp_storage;

public:
    BijectiveFunctionScanShuffle()
    {
        temp_storage.resize( 1 << 16 );
    }

    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        using ValueT = typename ContainerType::value_type;
        assert( &in_container != &out_container );

        RandomGenerator random_function( seed );
        BijectiveFunction mapping_function;
        mapping_function.init( num, random_function );
        uint64_t capacity = mapping_function.getMappingRange();

        thrust::counting_iterator<uint64_t> indexes( 0 );
        size_t m = num;
        auto input_iter = in_container.begin();
        auto mapping_it = thrust::make_transform_iterator( indexes, [=] __device__( uint64_t val ) -> ValueT {
            auto gather_key = mapping_function( val );
            if( gather_key < m )
            {
                return input_iter[gather_key];
            }
            return ValueT();
        } );
        auto flag_it = thrust::make_transform_iterator( indexes, [=] __device__( uint64_t val ) {
            auto gather_key = mapping_function( val );
            return gather_key < m;
        } );

        auto output_it = out_container.begin();
        // Determine temporary device storage requirements
        size_t temp_storage_bytes = 0;
        cub::DeviceSelect::Flagged( nullptr, temp_storage_bytes, mapping_it, flag_it, output_it,
                                    thrust::discard_iterator<int>(), capacity );
        // Allocate temporary storage
        if( temp_storage.size() < temp_storage_bytes )
        {
            temp_storage.resize( temp_storage_bytes );
        }
        // Run selection
        cub::DeviceSelect::Flagged( reinterpret_cast<void*>( temp_storage.data().get() ),
                                    temp_storage_bytes, mapping_it, flag_it, output_it,
                                    thrust::discard_iterator<int>(), capacity );
    }

    bool supportsInPlace() const override
    {
        return false;
    }
};