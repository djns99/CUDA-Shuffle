#pragma once
#include <thrust/device_vector.h>

#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"
#include <cub/cub.cuh>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>


struct IsValidFunctor
{
    int m;
    __forceinline__ IsValidFunctor( int m ) : m( m )
    {
    }
    __device__ __forceinline__ bool operator()( const uint64_t& x ) const
    {
        return x < m;
    }
};

template <class BijectiveFunction, class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class BijectiveFunctionScanShuffle : public Shuffle<ContainerType, RandomGenerator>
{
    thrust::device_vector<uint8_t> temp_storage;
    thrust::device_vector<uint64_t> gather_keys;

public:
    BijectiveFunctionScanShuffle()
    {
        temp_storage.resize( 1 << 16 );
        gather_keys.resize( 1 << 27 + 1 );
    }
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        assert( &in_container != &out_container );

        RandomGenerator random_function( seed );
        BijectiveFunction mapping_function;
        mapping_function.init( num, random_function );
        uint64_t capacity = mapping_function.getMappingRange();

        thrust::counting_iterator<uint64_t> indexes( 0 );
        auto mapping_it =
            thrust::make_transform_iterator( indexes, [mapping_function] __device__( uint64_t val ) -> uint64_t {
                // Call mapping functions
                return mapping_function( val );
            } );
        if( gather_keys.size() < in_container.size() )
        {
            gather_keys.resize( in_container.size() );
        }
        // Determine temporary device storage requirements
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceSelect::If( d_temp_storage, temp_storage_bytes, mapping_it, gather_keys.begin(),
                               thrust::discard_iterator<int>(), capacity,
                               IsValidFunctor( in_container.size() ) );
        // Allocate temporary storage
        if( temp_storage.size() < temp_storage_bytes )
        {
            temp_storage.resize( temp_storage_bytes );
        }
        // Run selection
        cub::DeviceSelect::If( reinterpret_cast<void*>( temp_storage.data().get() ), temp_storage_bytes,
                               mapping_it, gather_keys.begin(), thrust::discard_iterator<int>(),
                               capacity, IsValidFunctor( in_container.size() ) );
        thrust::gather( gather_keys.begin(), gather_keys.begin() + in_container.size(),
                        in_container.begin(), out_container.begin() );
    }

    bool supportsInPlace() override
    {
        return false;
    }
};