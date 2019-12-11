#pragma once
#include <thrust/device_vector.h>
#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"

__global__ void generatorInitKernel( GPURandomGenerator* gen, uint64_t count, uint64_t seed ) {
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if( tid >= count )
        return;

    gen[tid] = GPURandomGenerator( seed, tid );
}

template <class ElementType, uint64_t rounds_per_kernel, uint64_t initial_stride_log>
__global__ void gpuSwapKernel( ElementType* container, uint64_t count, GPURandomGenerator* generators )
{
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto& gen = generators[tid];
    uint64_t stride = 1ull << initial_stride_log;
    uint64_t offset = ( tid >> initial_stride_log ) * ( stride * 2 ) + ( tid & ( stride - 1 ) );
    for( uint64_t i = 0; i < rounds_per_kernel; i++ )
    {
        if( offset + stride >= count )
        {
            return;
        }

        bool swap = gen.getBool();
        if( swap )
        {
            uint64_t swap_idx = offset + stride;
            const auto temp = container[offset];
            container[offset] = container[swap_idx];
            container[swap_idx] = temp;
        }

        // Next offset can be defined by recursive relation
        // Tn = (tid & stride) != 0 ->  Tn-1 - stride
        //      otherwise              ->  Tn-1
        // Where stride = 2^n
        offset -= tid & stride;
        stride *= 2;
        __syncthreads();
    }
}

#define checkCudaError( ans )                           \
    {                                                   \
        assertCudaError( ( ans ), __FILE__, __LINE__ ); \
    }
inline void assertCudaError( cudaError_t code, std::string file, int line )
{
    if( code != cudaSuccess )
    {
        throw std::runtime_error( "CUDA Error " + std::string( cudaGetErrorString( code ) ) + " " +
                                  file + ":" + std::to_string( line ) );
    }
}


template <class ElementType = uint64_t>
class GPUSwapShuffle : public Shuffle<thrust::device_vector<ElementType>, GPURandomGenerator>
{
private:
    constexpr static uint64_t threads_per_block_log = 8;
    constexpr static uint64_t elements_per_block_log = threads_per_block_log + 1;
    constexpr static uint64_t threads_per_block = 1ull << threads_per_block_log;
    constexpr static uint64_t elements_per_block = 1ull << elements_per_block_log;

    constexpr static decltype(
        &gpuSwapKernel<ElementType, elements_per_block_log, 0> ) kernels[6] = {
        &gpuSwapKernel<ElementType, elements_per_block_log, elements_per_block_log * 0>,
        &gpuSwapKernel<ElementType, elements_per_block_log, elements_per_block_log * 1>,
        &gpuSwapKernel<ElementType, elements_per_block_log, elements_per_block_log * 2>,
        &gpuSwapKernel<ElementType, elements_per_block_log, elements_per_block_log * 3>,
        &gpuSwapKernel<ElementType, elements_per_block_log, elements_per_block_log * 4>,
        &gpuSwapKernel<ElementType, elements_per_block_log, elements_per_block_log * 5>,
    };
public:
    void shuffle( const thrust::device_vector<ElementType>& in_container,
                  thrust::device_vector<ElementType>& out_container,
                  uint64_t seed,
                  uint64_t num ) override
    {
        cudaStream_t stream;
        checkCudaError( cudaStreamCreate( &stream ) );
        const uint64_t blocks_per_kernel = ( num + elements_per_block - 1 ) / elements_per_block;

        GPURandomGenerator* d_generators;
        checkCudaError( cudaMalloc( &d_generators, num * sizeof(GPURandomGenerator) ) );
        std::unique_ptr<GPURandomGenerator[], std::function<void(GPURandomGenerator*)>> generators(d_generators, [](GPURandomGenerator* ptr) {
            checkCudaError( cudaFree( ptr ) );
        });
        generatorInitKernel<<<blocks_per_kernel, threads_per_block, 0, stream>>>( generators.get(), num, seed );

        if( &in_container != &out_container )
        {
            // Copy if we are not doing an inplace operation
            std::copy( in_container.begin(), in_container.begin() + num, out_container.begin() );
        }

        uint64_t stride = 1;
        for( auto& kernel : kernels )
        {
            kernel<<<blocks_per_kernel, threads_per_block, 0, stream>>>( thrust::raw_pointer_cast(
                                                                             out_container.data() ),
                                                                         num, generators.get() );
            checkCudaError( cudaGetLastError() );
            stride <<= elements_per_block_log;
            if( stride >= num )
                return;
        }
        checkCudaError( cudaStreamSynchronize( stream ) );
    }
};

template<class ElementType>
constexpr decltype(
        &gpuSwapKernel<ElementType, GPUSwapShuffle<ElementType>::elements_per_block_log, 0> ) GPUSwapShuffle<ElementType>::kernels[];