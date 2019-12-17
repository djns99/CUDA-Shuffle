#pragma once
#include "shuffle/BijectiveFunctionCompressor.h"
#include "shuffle/BijectiveFunctionScanShuffle.h"
#include "shuffle/BijectiveFunctionShuffle.h"
#include "shuffle/BijectiveFunctionSortShuffle.h"

template <uint64_t num_rounds>
class FeistelBijectiveFunction
{
public:
    template <class RandomGenerator>
    void init( uint64_t capacity, RandomGenerator& random_function )
    {
        side_bits = getCipherBits( capacity );
        side_mask = ( 1ull << side_bits ) - 1;
        for( uint64_t i = 0; i < num_rounds; i++ )
        {
            key[i] = (uint32_t)random_function();
        }
    }

    uint64_t getMappingRange() const
    {
        return 1ull << side_bits;
    }
    __host__ __device__ uint64_t operator()( const uint64_t val ) const
    {
        RoundState state = { ( uint32_t )( val >> side_bits ), ( uint32_t )( val & side_mask ) };

        for( uint64_t i = 0; i < num_rounds; i++ )
        {
            state = doRound( state, i );
        }

        return state.getValue( side_bits );
    }

private:
    uint64_t getCipherBits( uint64_t capacity )
    {
        // Have at least 8 bits worth of key for sbox
        if( capacity < 256 )
            return 4;
        uint64_t i = 0;
        while( capacity != 0 )
        {
            i++;
            capacity >>= 2;
        }
        return i;
    }

    struct RoundState
    {
        uint32_t left, right;

        __host__ __device__ uint64_t getValue( uint64_t side_bits ) const
        {
            return ( ( (uint64_t)left ) << side_bits ) | right;
        }
    };

    __host__ __device__ uint32_t applyKey( const uint32_t in_value, const uint32_t key ) const
    {
        uint32_t value = in_value ^ key;
        uint32_t new_val = 0;
        for( uint64_t i = 0; i <= side_bits - 4; i += 4 )
        {
            new_val = ( new_val << 4 ) | sbox16( ( value >> i ) & 0xF );
        }
        return new_val & side_mask;
    }

    // __host__ __device__ uint32_t applyKey( uint64_t value, const uint64_t key ) const
    // {
    //     value ^= key;
    //     for( uint64_t i = 0; i < 5; i++ )
    //     {
    //         uint64_t key_region = ( key >> ( i * 12 ) ) & 0xFFF;
    //         uint64_t shift1 = key_region & 0x3F;
    //         uint64_t shift2 = key_region >> 6;
    //         value ^= sbox256[( value >> shift1 ) & 0xFF] << shift2;
    //     }
    //     for( uint64_t i = 0; i < 64; i += side_bits )
    //     {
    // value ^= (value >> i) & side_mask;
    //     }
    //     return value & side_mask;
    // }

    //  __host__ __device__ uint32_t applyKey( uint64_t value, const uint64_t key[3] ) const
    //  {
    //      value ^= value >> 12;
    //      value ^= value << 25;
    //      value ^= value >> 27;
    //// Initialise u,v,w for random number generator
    //      uint64_t u = value ^ key[0];
    //      value ^= value >> 12;
    //      value ^= value << 25;
    //      value ^= value >> 27;
    //      uint64_t v = value ^ key[1];
    //      value ^= value >> 12;
    //      value ^= value << 25;
    //      value ^= value >> 27;
    //      uint64_t w = value ^ key[2];
    //      u = u * 2862933555777941757LL + 7046029254386353087LL;
    //      v ^= v >> 17;
    //      v ^= v << 31;
    //      v ^= v >> 8;
    //      w = 4294957665U * ( w & 0xffffffff ) + ( w >> 32 );
    //      uint64_t x = u ^ ( u << 21 );
    //      x ^= x >> 35;
    //      x ^= x << 4;
    //      return (( x + v ) ^ w) & side_mask;
    //  }

    __host__ __device__ RoundState doRound( const RoundState state, const uint64_t round ) const
    {
        uint32_t new_left = state.right;
        uint32_t new_right = state.left ^ applyKey( state.right, key[round] );
        return { new_left, new_right };
    }

    uint64_t side_bits;
    uint64_t side_mask;
    uint32_t key[num_rounds];


    __host__ __device__ uint8_t sbox16( uint8_t index ) const
    {
        // Random 4-bit s-box
        static const uint8_t sbox16[16] = { 0xC, 0x5, 0x0, 0xE, 0x9, 0x3, 0xA, 0xD,
                                            0xB, 0x6, 0x7, 0x8, 0x4, 0xF, 0x1, 0x2 };
        return sbox16[index];
    }
};

static constexpr uint64_t FEISTEL_DEFAULT_NUM_ROUNDS = 12;
template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using FeistelBijectiveShuffle =
    BijectiveFunctionShuffle<BijectiveFunctionCompressor<FeistelBijectiveFunction<FEISTEL_DEFAULT_NUM_ROUNDS>>, ContainerType, RandomGenerator>;

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using FeistelBijectiveSortShuffle =
    BijectiveFunctionSortShuffle<FeistelBijectiveFunction<FEISTEL_DEFAULT_NUM_ROUNDS>, ContainerType, RandomGenerator>;

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using FeistelBijectiveScanShuffle =
    BijectiveFunctionScanShuffle<FeistelBijectiveFunction<FEISTEL_DEFAULT_NUM_ROUNDS>, ContainerType, RandomGenerator>;