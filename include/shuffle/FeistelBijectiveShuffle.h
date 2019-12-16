#pragma once
#include "shuffle/BijectiveFunctionShuffle.h"
#include "shuffle/BijectiveFunctionSortShuffle.h"
#include "shuffle/BijectiveFunctionCompressor.h"

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
        static const uint8_t sbox16[16] = { 0x7, 0x5, 0xD, 0x8, 0x9, 0xA, 0xF, 0x3,
                                            0x6, 0x1, 0xE, 0x0, 0xC, 0x4, 0x2, 0xB };
        return sbox16[index];
    }

    __host__ __device__ uint8_t sbox256( uint8_t index ) const
    {
        static const uint8_t sbox256[256] = {
            0x55, 0x95, 0x5d, 0x1c, 0x60, 0xba, 0x0b, 0xfb, 0x67, 0xbd, 0x4d, 0xef, 0xa0, 0xb3,
            0x9b, 0x1f, 0xe8, 0x1a, 0xc1, 0x41, 0xc3, 0x4f, 0x34, 0x25, 0xb9, 0xb6, 0xb1, 0x87,
            0xf2, 0x3d, 0x5b, 0x08, 0x76, 0x6f, 0xbb, 0x51, 0x86, 0xe9, 0xde, 0x2f, 0x88, 0x1d,
            0x68, 0xae, 0x26, 0x0c, 0xad, 0x4e, 0xab, 0x03, 0xa5, 0x74, 0x31, 0x6e, 0xf5, 0x4b,
            0xcc, 0x66, 0x6c, 0xe1, 0xc4, 0x78, 0x5c, 0x1e, 0x36, 0x0a, 0x13, 0x14, 0x64, 0x24,
            0xac, 0xb7, 0x4c, 0x38, 0xfd, 0xb8, 0xb0, 0x7b, 0x8f, 0x43, 0x93, 0x05, 0xdc, 0xd9,
            0x54, 0x39, 0x0f, 0x75, 0x9e, 0xce, 0x0e, 0xd5, 0x8e, 0xe7, 0x80, 0x94, 0xed, 0x47,
            0x99, 0xa1, 0x82, 0x97, 0x21, 0x8d, 0x2b, 0x52, 0xcd, 0x83, 0x12, 0xee, 0x92, 0x45,
            0x53, 0x0d, 0x16, 0xf7, 0x2c, 0x2a, 0xfa, 0x10, 0xb2, 0xbe, 0x33, 0x07, 0xe3, 0x7d,
            0x22, 0x04, 0xd3, 0xe5, 0x8a, 0xc6, 0xff, 0x2d, 0x69, 0x9a, 0x19, 0x5e, 0x27, 0x62,
            0xc8, 0xa6, 0xa7, 0xf0, 0x02, 0xec, 0xd8, 0xa9, 0x5a, 0xbf, 0x81, 0xcb, 0x23, 0x44,
            0x9c, 0xf1, 0x30, 0xd0, 0xa4, 0xa3, 0xe4, 0xf9, 0x72, 0x49, 0xca, 0x20, 0x96, 0x71,
            0x73, 0x7a, 0xd2, 0xf8, 0xe6, 0x59, 0x46, 0x09, 0x42, 0x17, 0xb4, 0x8b, 0x11, 0x18,
            0xc2, 0x3c, 0x48, 0xa8, 0xc0, 0xa2, 0x15, 0x56, 0xd1, 0x4a, 0x37, 0x63, 0xb5, 0xf4,
            0x65, 0x85, 0xfe, 0x7e, 0x06, 0x40, 0x91, 0x77, 0x3b, 0xcf, 0xc7, 0x2e, 0x00, 0x50,
            0x84, 0xdb, 0xea, 0xd4, 0x90, 0xd7, 0x3e, 0x29, 0xe2, 0xbc, 0xc9, 0x7f, 0x28, 0xaa,
            0x57, 0xda, 0x89, 0x1b, 0x8c, 0x3f, 0xf6, 0x6a, 0xdf, 0xf3, 0xe0, 0xdd, 0xd6, 0x9f,
            0x58, 0x6d, 0x9d, 0x01, 0xeb, 0xfc, 0x61, 0x98, 0x32, 0xc5, 0x3a, 0x5f, 0x70, 0x35,
            0xaf, 0x7c, 0x6b, 0x79,
        };
        return sbox256[index];
    }
};

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using FeistelBijectiveShuffle =
    BijectiveFunctionShuffle<BijectiveFunctionCompressor<FeistelBijectiveFunction<8>>, ContainerType, RandomGenerator>;

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using FeistelBijectiveSortShuffle =
    BijectiveFunctionSortShuffle<FeistelBijectiveFunction<8>, ContainerType, RandomGenerator>;