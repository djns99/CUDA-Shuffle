#pragma once
#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/device_vector.h>


struct KeyFlagTuple
{
    uint64_t key;
    uint64_t flag;
};

struct ScanOp
{
    __device__ KeyFlagTuple operator()( const KeyFlagTuple& a, const KeyFlagTuple& b )
    {
        return { b.key, a.flag + b.flag };
    }
};

template <class BijectiveFunction>
struct MakeTupleFunctor
{
    uint64_t m;
    BijectiveFunction mapping_function;
    MakeTupleFunctor( uint64_t m, BijectiveFunction mapping_function )
        : m( m ), mapping_function( mapping_function )
    {
    }
    __device__ KeyFlagTuple operator()( uint64_t idx )
    {
        auto gather_key = mapping_function( idx );
        return KeyFlagTuple{ gather_key, gather_key < m };
    }
};

template <typename InputIterT, typename OutputIterT>
struct WritePermutationFunctor
{
    uint64_t m;
    InputIterT in;
    OutputIterT out;
    __device__ size_t operator()( KeyFlagTuple x )
    {
        if( x.key < m )
        {
            // -1 because inclusive scan
            out[x.flag - 1] = in[x.key];
        }
        return 0; // Discarded
    }
};

struct cached_allocator
{
    typedef char value_type;

    thrust::device_vector<char> memory;

    char* allocate( std::ptrdiff_t num_bytes )
    {
        memory.resize( num_bytes );
        return memory.data().get();
    }

    void deallocate( char* ptr, size_t )
    {
    }
};

template <class BijectiveFunction, class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class BijectiveFunctionScanShuffle : public Shuffle<ContainerType, RandomGenerator>
{
    cached_allocator alloc;

public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        assert( &in_container != &out_container );

        RandomGenerator random_function( seed );
        BijectiveFunction mapping_function;
        mapping_function.init( num, random_function );
        uint64_t capacity = mapping_function.getMappingRange();

        thrust::counting_iterator<uint64_t> indices( 0 );
        size_t m = num;
        thrust::transform_iterator<MakeTupleFunctor<BijectiveFunction>, decltype( indices ), KeyFlagTuple> tuple_it(
            indices, MakeTupleFunctor<BijectiveFunction>( m, mapping_function ) );
        WritePermutationFunctor<decltype( in_container.begin() ), decltype( out_container.begin() )> write_functor{
            m, in_container.begin(), out_container.begin()
        };
        auto output_it =
            thrust::make_transform_output_iterator( thrust::discard_iterator<size_t>(), write_functor );
        thrust::inclusive_scan( thrust::cuda::par( alloc ), tuple_it, tuple_it + capacity, output_it, ScanOp() );
    }

    bool supportsInPlace() const override
    {
        return false;
    }
};