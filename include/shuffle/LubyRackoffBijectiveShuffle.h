#pragma once
#include "shuffle/BijectiveFunctionCompressor.h"
#include "shuffle/BijectiveFunctionScanShuffle.h"
#include "shuffle/BijectiveFunctionShuffle.h"
#include "shuffle/BijectiveFunctionSortShuffle.h"
#include <cuda_runtime.h>

class LubyRackoffBijectiveFunction
{
private:
    constexpr static uint64_t num_rounds = 4;
    struct Key
    {
        uint64_t wyhash[4];
        uint64_t lehmer[4][2];
    };
    struct RoundState
    {
        uint32_t left, right;

        __device__ uint64_t getValue( uint64_t side_bits ) const
        {
            return ( ( (uint64_t)left ) << side_bits ) | right;
        }
    };

public:
    template <class RandomGenerator>
    void init( uint64_t capacity, RandomGenerator& random_function )
    {
        side_bits = getCipherBits( capacity );
        side_mask = ( 1ull << side_bits ) - 1;
        for( uint64_t i = 0; i < num_rounds; i++ )
        {
            key[i] = { { random_function(), random_function(), random_function(), random_function() },
                       { { random_function(), random_function() },
                         { random_function(), random_function() },
                         { random_function(), random_function() },
                         { random_function(), random_function() } } };
        }
    }

    uint64_t getMappingRange() const
    {
        return 1ull << side_bits;
    }

    __device__ uint64_t operator()( const uint64_t val ) const
    {
        RoundState state = { ( uint32_t )( val >> side_bits ), ( uint32_t )( val & side_mask ) };
        state = lubyRackoff( state );
        return state.getValue( side_bits );
    }

private:
    /*
     * wyhash64 hash function
     */
    __device__ uint64_t wyhash64( uint32_t input, uint64_t& wyhash64 ) const
    {
        wyhash64 ^= input;
        wyhash64 += 0x60bee2bee120fc15;
        uint64_t x = wyhash64;
        constexpr uint64_t y = 0xa3b195354a39b70d;
        constexpr uint64_t z = 0x1b03738712fad5c9;
        uint64_t w = __mul64hi( x, y ) ^ ( x * y );
        return __mul64hi( z, w ) ^ ( z * w );
    }

    /*
     * lehmer64 hash function
     */
    __device__ uint64_t lehmer64( uint32_t input, uint64_t& key_lo, uint64_t& key_hi ) const
    {
        key_lo ^= input;
        key_hi ^= ~input;
        uint64_t y = 0xda942042e4dd58b5;
        key_hi += __mul64hi( key_lo, y );
        key_lo *= y;
        return key_hi;
    }

    __device__ uint64_t prng( uint32_t input, uint64_t round, Key& key ) const
    {
        const uint64_t wyhash = wyhash64( input, key.wyhash[round & 3] );
        const uint64_t lehmer = lehmer64( input, key.lehmer[round & 3][0], key.lehmer[round & 3][1] );
        return wyhash ^ lehmer;
    }

    /*
     * GGM Works by using an n -> 2n bit prbg and applying it once for every bit in the input
     * If the bit is 1 the upper n bits are used, otherwise the lower n bits are used
     */
    __device__ uint32_t ggmPRF( uint32_t state, Key key ) const
    {
        uint32_t ggm_state = state;
        for( uint64_t i = 0; i < side_bits; i++, state >>= 1 )
        {
            ggm_state = ( prng( ggm_state, i, key ) >> ( side_bits * ( state & 1 ) ) ) & side_mask;
        }
        return ggm_state;
    }

    /*
     *  Luby-Rackoff Construction is a four round feistel network that uses a PRF
     */
    __device__ RoundState lubyRackoff( RoundState state ) const
    {
        const uint32_t alpha = ggmPRF( state.right, key[0] );
        const uint32_t beta = ggmPRF( state.left ^ alpha, key[1] );
        const uint32_t gamma = ggmPRF( state.right ^ beta, key[2] );
        const uint32_t delta = ggmPRF( state.left ^ alpha ^ gamma, key[3] );
        return { state.left ^ alpha ^ gamma, state.right ^ beta ^ delta };
    }

    uint64_t getCipherBits( uint64_t capacity )
    {
        uint64_t i = 0;
        while( capacity != 0 )
        {
            i++;
            capacity >>= 2;
        }
        return i;
    }

    uint64_t side_bits;
    uint64_t side_mask;
    Key key[num_rounds];
};

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using LubyRackoffBijectiveShuffle =
    BijectiveFunctionShuffle<BijectiveFunctionCompressor<LubyRackoffBijectiveFunction>, ContainerType, RandomGenerator>;

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using LubyRackoffBijectiveSortShuffle =
    BijectiveFunctionSortShuffle<LubyRackoffBijectiveFunction, ContainerType, RandomGenerator>;

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using LubyRackoffBijectiveScanShuffle =
    BijectiveFunctionScanShuffle<LubyRackoffBijectiveFunction, ContainerType, RandomGenerator>;