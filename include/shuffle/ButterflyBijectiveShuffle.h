#pragma once
#include "BijectiveFunctionCompressor.h"
#include "BijectiveFunctionScanShuffle.h"
#include "BijectiveFunctionShuffle.h"
#include "BijectiveFunctionSortShuffle.h"
#include "ThrustInclude.h"
#include <mutex>

__global__ void initRandom( uint64_t words, uint64_t* data, GPURandomGenerator* generators, uint64_t current_gens, uint64_t seed )
{
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if( tid >= words )
        return;
    if( tid >= current_gens )
        new( generators + tid ) GPURandomGenerator( seed, tid );
    data[tid] = generators[tid]();
}

class ButterflyAllocator
{
public:
    uint64_t resize( uint64_t items )
    {
        uint64_t old_size = bits_cache.size();
        if( items > bits_cache.size() )
        {
            bits_cache.resize( items );
            gen_cache.resize( items );
        }
        return old_size;
    }

    ~ButterflyAllocator()
    {
        assert( cudaDeviceSynchronize() == cudaSuccess );
    }

    // TODO Add param for CPU
    thrust::device_vector<uint64_t> bits_cache;
    thrust::device_vector<GPURandomGenerator> gen_cache;
};

class ButterflyNetworkBijection
{
public:
    template <class RandomGenerator>
    void init( uint64_t count, RandomGenerator& random_function, ButterflyAllocator& allocator )
    {
        m = log2( count );
        rand_words = getRandWords( m );

        uint64_t old_size = allocator.resize( rand_words );
        gpu_bits_ptr = allocator.bits_cache.data().get();
        gpu_generator_ptr = allocator.gen_cache.data().get();

        const uint64_t blocks = ( rand_words + 255 ) / 256;
        initRandom<<<256, blocks>>>( rand_words, gpu_bits_ptr, gpu_generator_ptr, old_size, random_function() );
    }

    uint64_t getMappingRange()
    {
        return 1ull << m;
    }

    __host__ __device__ uint64_t operator()( uint64_t index ) const
    {
        if( m == 0 )
            return 0;
        const uint64_t bits_per_pass = 1ull << ( m - 1 );
        for( uint64_t i = 0; i < m; i++ )
        {
            // Calculate which items we will be switching
            const uint64_t stride = 1ull << i;
            const uint64_t mod = index & ( stride - 1 );
            const uint64_t div = index >> i;

            // Calculate which bit to read to tell if we swap
            // Ignore the lowest bit of div so both sides of switch read the same bit
            const uint64_t pass_bit = ( div >> 1 ) * stride + mod;
            const uint64_t abs_bit_loc = i * bits_per_pass + pass_bit;
            const uint64_t byte_idx = abs_bit_loc / 8;
            const uint64_t byte_off = abs_bit_loc % 8;

            // Read the bit and swap the items
            // Lowest bit of the divisor indicates which end of the swap the index is
            const auto* const bits = (uint8_t*)gpu_bits_ptr;
            const uint64_t sel = ( bits[byte_idx] >> byte_off ) & 1;
            const uint64_t new_div = div ^ sel;

            index = ( new_div << i ) + mod;
        }

        return index;
    }

private:
    static uint64_t log2( uint64_t capacity )
    {
        if( capacity == 0 )
            return 0;
        uint64_t i = 0;
        capacity--;
        while( capacity != 0 )
        {
            i++;
            capacity >>= 1;
        }
        return i;
    }

    static uint64_t getRandBits( uint64_t m )
    {
        if( m == 0 )
            return 0;
        // n = 2^m
        // (n/2) * log (n/2) swaps
        m--;
        return ( 1ull << m ) * m;
    }

    static uint64_t getRandWords( uint64_t m )
    {
        return ( getRandBits( m ) + 63 ) / 64;
    }

    uint64_t m = 0;
    uint64_t rand_words = 0;
    uint64_t* gpu_bits_ptr = nullptr;
    GPURandomGenerator* gpu_generator_ptr = nullptr;
};

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class ButterflyNetworkBijectiveSortShuffle : public Shuffle<ContainerType, RandomGenerator>
{
private:
    thrust::device_vector<uint8_t> temp_storage;
    thrust::device_vector<uint64_t> key_in;
    thrust::device_vector<uint64_t> key_out;

public:
    ButterflyNetworkBijectiveSortShuffle()
    {
        temp_storage.resize( 1 << 16 );
        key_in.resize( 1 << 16 );
        key_out.resize( 1 << 16 );
    }

    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        RandomGenerator random_function( seed );
        ButterflyNetworkBijection mapping_function;
        mapping_function.init( num, random_function, allocator );

        if( num > key_in.size() )
        {
            key_in.resize( num );
            key_out.resize( num );
        }

        // Initialise key vector with indexes
        thrust::counting_iterator<uint64_t> counting_begin( 0 );
        auto counting_end = counting_begin + num;
        // Inplace transform
        thrust::transform( thrust::device, counting_begin, counting_end, key_in.begin(),
                           [mapping_function] __host__ __device__( uint64_t val ) -> uint64_t {
                               return mapping_function( val );
                           } );

        // Determine temporary device storage requirements
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs( NULL, temp_storage_bytes, key_in.data().get(),
                                         key_out.data().get(), in_container.data().get(),
                                         out_container.data().get(), num );

        if( temp_storage_bytes > temp_storage.size() )
        {
            temp_storage.resize( temp_storage_bytes );
        }

        cub::DeviceRadixSort::SortPairs( temp_storage.data().get(), temp_storage_bytes,
                                         key_in.data().get(), key_out.data().get(),
                                         in_container.data().get(), out_container.data().get(), num );
    }

    bool supportsInPlace() const override
    {
        return false;
    }

    bool isDeterministic() const override
    {
        return false;
    }

private:
    ButterflyAllocator allocator;
};

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class ButterflyNetworkBijectiveScanShuffle : public Shuffle<ContainerType, RandomGenerator>
{
    cached_allocator alloc;

    // thrust::device_vector<KeyFlagTuple> result;

public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        assert( &in_container != &out_container );

        RandomGenerator random_function( seed );
        ButterflyNetworkBijection mapping_function;
        mapping_function.init( num, random_function, allocator );
        uint64_t capacity = mapping_function.getMappingRange();

        thrust::counting_iterator<uint64_t> indices( 0 );
        size_t m = num;
        thrust::transform_iterator<MakeTupleFunctor<ButterflyNetworkBijection>, decltype( indices ), KeyFlagTuple> tuple_it(
            indices, MakeTupleFunctor<ButterflyNetworkBijection>( m, mapping_function ) );
        WritePermutationFunctor<decltype( in_container.begin() ), decltype( out_container.begin() )> write_functor{
            m, in_container.begin(), out_container.begin()
        };
        auto output_it =
            thrust::make_transform_output_iterator( thrust::discard_iterator<uint64_t>(), write_functor );
        thrust::inclusive_scan( thrust::cuda::par( alloc ), tuple_it, tuple_it + capacity, output_it, ScanOp() );

        // Without transform output iterator
        // if( result.size() < capacity )
        // {
        //     result.resize( capacity );
        // }
        // thrust::inclusive_scan( thrust::cuda::par( alloc ), tuple_it, tuple_it + capacity,
        //                         result.begin(), ScanOp() );
        // thrust::for_each( result.begin(), result.begin() + capacity, write_functor );
    }

    bool supportsInPlace() const override
    {
        return false;
    }

    bool isDeterministic() const override
    {
        return false;
    }

private:
    ButterflyAllocator allocator;
};