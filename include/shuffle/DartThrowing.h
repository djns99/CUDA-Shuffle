#pragma once
#include "DefaultRandomGenerator.h"
#include "ThrustInclude.h"
#include "shuffle/Shuffle.h"
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>

__global__ void dartThrowingKernel( uint64_t* indices, uint64_t num, uint64_t targets, uint64_t seed )
{
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if( tid >= num )
        return;

    GPURandomGenerator rng( seed, tid );

    uint64_t index;
    do
    {
        index = rng() % targets;
    } while( ( tid = atomicExch( (unsigned long long*)indices + index, (unsigned long long)tid ) ) != UINT64_MAX );
}

namespace DartThrowingScanFuncs
{
struct KeyFlagTuple
{
    uint64_t key;
    uint64_t flag;
};

struct ScanOp
{
    __host__ __device__ KeyFlagTuple operator()( const KeyFlagTuple& a, const KeyFlagTuple& b )
    {
        return { b.key, a.flag + b.flag };
    }
};

struct MakeTupleFunctor
{
    __host__ __device__ KeyFlagTuple operator()( uint64_t idx )
    {
        return KeyFlagTuple{ idx, idx != UINT64_MAX };
    }
};

template <typename InputIterT, typename OutputIterT>
struct WritePermutationFunctor
{
    uint64_t m;
    InputIterT in;
    OutputIterT out;
    __host__ __device__ size_t operator()( KeyFlagTuple x )
    {
        if( x.key != UINT64_MAX )
        {
            // flag - 1 because inclusive scan
            out[x.flag - 1] = in[x.key];
        }
        return 0; // Discarded
    }
};
}

template <class ContainerType = thrust::device_vector<uint64_t>, uint64_t alpha_numerator = 4, uint64_t alpha_denom = 1>
class DartThrowing : public Shuffle<ContainerType, GPURandomGenerator>
{
public:
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

    virtual void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        using namespace DartThrowingScanFuncs;
        const uint64_t num_targets = num * (double)alpha_numerator / (double)alpha_denom + 1;
        if( temp_storage.size() < num_targets )
        {
            temp_storage.resize( num_targets );
        }
        thrust::fill( thrust::cuda::par, temp_storage.begin(), temp_storage.begin() + num_targets, UINT64_MAX );

        dartThrowingKernel<<<( num + 255 ) / 256, 256>>>( temp_storage.data().get(), num, num_targets, seed );

        thrust::transform_iterator<MakeTupleFunctor, decltype( temp_storage.begin() ), KeyFlagTuple> tuple_it(
            temp_storage.begin(), MakeTupleFunctor{} );
        WritePermutationFunctor<decltype( in_container.begin() ), decltype( out_container.begin() )> write_functor{
            num, in_container.begin(), out_container.begin()
        };
        auto output_it =
            thrust::make_transform_output_iterator( thrust::discard_iterator<uint64_t>(), write_functor );
        thrust::inclusive_scan( thrust::cuda::par( alloc ), tuple_it, tuple_it + num_targets,
                                output_it, ScanOp() );
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

template <class ContainerType = thrust::host_vector<uint64_t>, uint64_t alpha_numerator = 4, uint64_t alpha_denom = 1>
class HostDartThrowing : public Shuffle<ContainerType, DefaultRandomGenerator>
{
public:
    struct cached_allocator
    {
        typedef char value_type;

        thrust::host_vector<char> memory;

        char* allocate( std::ptrdiff_t num_bytes )
        {
            memory.resize( num_bytes );
            return memory.data();
        }

        void deallocate( char* ptr, size_t )
        {
        }
    };

    cached_allocator alloc;
    thrust::host_vector<uint64_t> temp_storage;
    uint64_t storage_size = 0;

    void dartThrowingHost( thrust::host_vector<uint64_t>& indices,
                           uint64_t start,
                           uint64_t end,
                           uint64_t num_targets,
                           DefaultRandomGenerator& generator )
    {
        std::uniform_int_distribution<uint64_t> dist( 0, num_targets - 1 );

        for( uint64_t i = start; i < end; i++ )
        {
            uint64_t index = i;
            do
            {
                // TODO Make this work for MSVC
                uint64_t target = dist( generator );
                assert( target < num_targets );
                __atomic_exchange( indices.data() + target, &index, &index, __ATOMIC_RELAXED );
                assert( index < num_targets || index == UINT64_MAX );
            } while( index != UINT64_MAX );
        }
    }

    virtual void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        using namespace DartThrowingScanFuncs;

        const uint64_t num_targets = num * (double)alpha_numerator / (double)alpha_denom + 1;
        if( temp_storage.size() < num_targets )
        {
            temp_storage.resize( num_targets );
        }
        std::fill( temp_storage.begin(), temp_storage.begin() + num_targets, UINT64_MAX );

        uint64_t num_threads = std::thread::hardware_concurrency();
        DefaultRandomGenerator seeder( seed );
        std::vector<DefaultRandomGenerator> generators( num_threads, seeder );

        const uint64_t num_per_thread = ( num + num_threads - 1 ) / num_threads;
        std::vector<std::thread> threads;
        for( uint64_t i = 0; i < num_threads; i++ )
        {
            threads.emplace_back( [this, i, num_per_thread, num, num_targets, &generators]() {
                dartThrowingHost( this->temp_storage, i * num_per_thread,
                                  std::min( num, ( i + 1 ) * num_per_thread ), num_targets, generators[i] );
            } );
        }

        for( auto& thread : threads )
            thread.join();

        // Ensure atomic operations are all resolved
        std::atomic_thread_fence( std::memory_order_acq_rel );

        thrust::transform_iterator<MakeTupleFunctor, decltype( temp_storage.begin() ), KeyFlagTuple> tuple_it(
            temp_storage.begin(), MakeTupleFunctor{} );
        WritePermutationFunctor<decltype( in_container.begin() ), decltype( out_container.begin() )> write_functor{
            num, in_container.begin(), out_container.begin()
        };
        auto output_it =
            thrust::make_transform_output_iterator( thrust::discard_iterator<uint64_t>(), write_functor );
        thrust::inclusive_scan( thrust::cpp::par( alloc ), tuple_it, tuple_it + num_targets,
                                output_it, ScanOp() );
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