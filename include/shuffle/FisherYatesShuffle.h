#pragma once
#include <vector>

#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"

template <class ContainerType = std::vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class FisherYatesShuffle : public Shuffle<ContainerType, RandomGenerator>
{
private:
    void swap( ContainerType& container, uint64_t a, uint64_t b )
    {
        auto temp = container[a];
        container[a] = container[b];
        container[b] = temp;
    }

public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        if( &in_container != &out_container )
        {
            // Copy if we are not doing an inplace operation
            std::copy( in_container.begin(), in_container.begin() + num, out_container.begin() );
        }

        RandomGenerator random_function( seed );
        for( uint64_t i = 0; i < num - 1; i++ )
        {
            auto swap_range = num - i;
            swap( out_container, i, i + ( random_function() % swap_range ) );
        }
    }
};

#include <thrust/device_vector.h>

template <class ElementType, class RandomGenerator, uint64_t rounds_per_kernel, uint64_t initial_stride_log>
__global__ void fisherYatesKernel( ElementType* container, uint64_t count, uint64_t seed )
{
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    RandomGenerator gen( seed + tid );
    uint64_t stride = 1ull << initial_stride_log;
    uint64_t offset = ( tid >> initial_stride_log ) * ( stride * 2 ) + ( tid & ( stride - 1 ) );
    if( offset + stride >= count )
    {
        return;
    }
    for( uint64_t i = 0; i < rounds_per_kernel; i++ )
    {
        bool swap = gen() % 2 == 0;
        if( swap )
        {
            uint64_t swap_idx = offset + stride;
            auto temp = container[offset];
            container[offset] = container[swap_idx];
            container[swap_idx] = temp;
        }

        // Next offset can be defined by recursive relation
        // Tn = (offset & stride) != 0 ->  Tn-1 - stride
        //      otherwise              ->  Tn-1
		// Where stride = 2^n
        offset -= offset & stride;
        stride *= 2;
        __syncthreads();
    }
}

#define checkCudaError( ans )								\
    {														\
        assertCudaError( ( ans ), __FILE__, __LINE__ );		\
    }
inline void assertCudaError( cudaError_t code, std::string file, int line )
{
    if( code != cudaSuccess )
    {
        throw std::runtime_error( "CUDA Error " + std::string( cudaGetErrorString( code ) ) + " " + file + ":" + std::to_string( line )  );
    }
}


template <class ElementType = uint64_t, class RandomGenerator = DefaultRandomGenerator>
class FisherYatesShuffleGPU : public Shuffle<thrust::device_vector<ElementType>, RandomGenerator>
{
public:
    void shuffle( const thrust::device_vector<ElementType>& in_container,
                  thrust::device_vector<ElementType>& out_container,
                  uint64_t seed,
                  uint64_t num ) override
    {
        if( &in_container != &out_container )
        {
            // Copy if we are not doing an inplace operation
            std::copy( in_container.begin(), in_container.begin() + num, out_container.begin() );
        }

        RandomGenerator random_function( seed );
        cudaStream_t stream;
        checkCudaError( cudaStreamCreate( &stream ) );
        const uint64_t blocks_per_kernel = (num + elements_per_block - 1) / elements_per_block;
        uint64_t stride = 1;
        for( auto kernel : kernels )
        {
            kernel<<<blocks_per_kernel, threads_per_block, 0, stream>>>( thrust::raw_pointer_cast(out_container.data()), num, random_function() );
            checkCudaError( cudaGetLastError() );
            stride <<= elements_per_block_log;
            if( stride >= num )
                return;
        }
        checkCudaError( cudaStreamSynchronize( stream ) );
    }

private:
    constexpr static uint64_t threads_per_block_log = 8;
    constexpr static uint64_t elements_per_block_log = threads_per_block_log + 1;
    constexpr static uint64_t threads_per_block = 1ull << threads_per_block_log;
    constexpr static uint64_t elements_per_block = 1ull << elements_per_block_log;

    constexpr static decltype(
        &fisherYatesKernel<ElementType, DefaultRandomGenerator, elements_per_block_log, 0> ) kernels[] = {
        &fisherYatesKernel<ElementType, DefaultRandomGenerator, elements_per_block_log, elements_per_block_log * 0>,
        &fisherYatesKernel<ElementType, DefaultRandomGenerator, elements_per_block_log, elements_per_block_log * 1>,
        &fisherYatesKernel<ElementType, DefaultRandomGenerator, elements_per_block_log, elements_per_block_log * 2>,
        &fisherYatesKernel<ElementType, DefaultRandomGenerator, elements_per_block_log, elements_per_block_log * 3>,
        &fisherYatesKernel<ElementType, DefaultRandomGenerator, elements_per_block_log, elements_per_block_log * 4>,
        &fisherYatesKernel<ElementType, DefaultRandomGenerator, elements_per_block_log, elements_per_block_log * 5>,
    };
};
