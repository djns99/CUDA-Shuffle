#pragma once
#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"
#include <cuda.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#if( __CUDA_API_VERSION >= 10000 )
#include <thrust/iterator/transform_output_iterator.h>
#endif
#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

__global__ void dartThrowing( uint64_t* bitmap, uint64_t num, uint64_t targets, uint64_t seed )
{
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if( tid >= num )
        return;

    GPURandomGenerator rng( seed, tid );

    uint64_t index;
    do
    {
        index = rng() % targets;
    } while( atomicCAS( (unsigned long long*)bitmap + index, 0ull, (unsigned long long)tid + 1 ) != 0 );
}

template <class ContainerType = thrust::device_vector<uint64_t>, uint64_t alpha_numerator = 3, uint64_t alpha_denom = 2>
class DartThrowing : public Shuffle<ContainerType, GPURandomGenerator>
{
public:
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

    struct MakeTupleFunctor
    {
        __device__ KeyFlagTuple operator()( uint64_t idx )
        {
            return KeyFlagTuple{ idx, idx != 0 };
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
            if( x.key != 0 )
            {
                // flag - 1 because inclusive scan
                out[x.flag - 1] = in[x.key - 1];
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

    cached_allocator alloc;
    thrust::device_vector<uint64_t> temp_storage;
    thrust::device_vector<KeyFlagTuple> result;

    virtual void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        const uint64_t num_targets = num * (double)alpha_numerator / (double)alpha_denom + 1;
        if( temp_storage.size() < num_targets )
        {
            temp_storage.resize( num_targets );
            result.resize( num_targets );
        }

        dartThrowing<<<( num + 255 ) / 256, 256>>>( temp_storage.data().get(), num, num_targets, seed );

        thrust::transform_iterator<MakeTupleFunctor, decltype( temp_storage.begin() ), KeyFlagTuple> tuple_it( temp_storage.begin(),
                                                                                       MakeTupleFunctor{} );
        WritePermutationFunctor<decltype( in_container.begin() ), decltype( out_container.begin() )> write_functor{
            num, in_container.begin(), out_container.begin()
        };
#if( __CUDA_API_VERSION >= 10000 )
        auto output_it =
            thrust::make_transform_output_iterator( thrust::discard_iterator<uint64_t>(), write_functor );
        thrust::inclusive_scan( thrust::cuda::par( alloc ), tuple_it, tuple_it + num_targets, output_it, ScanOp() );
#else
        thrust::inclusive_scan( thrust::cuda::par( alloc ), tuple_it, tuple_it + num_targets, result.begin(), ScanOp() );
        thrust::for_each( result.begin(), result.begin() + num_targets, write_functor );
#endif
    }

    bool supportsInPlace() const override
    {
        return false;
    }

    bool isDeterministic() const override
    {
        return false;
    }
};