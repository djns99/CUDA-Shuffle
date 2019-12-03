#pragma once
#include "shuffle/BijectiveFunctionShuffle.h"

template <uint64_t num_rounds>
class SPNetworkBijectiveFunction
{
public:
    template <class RandomGenerator>
    void init( uint64_t capacity, RandomGenerator& random_function )
    {
        this->capacity = capacity;
        num_bits = getCipherBits( capacity );
        bit_mask = ( 1ull << num_bits ) - 1;
        for( uint64_t i = 0; i < num_rounds; i++ )
            key[i] = random_function() & bit_mask;
    }

    __host__ __device__ uint64_t operator()( const uint64_t val ) const
    {
        uint64_t state = val;
        do
        {
            for( uint64_t i = 0; i < num_rounds; i++ )
            {
                state = doRound( state, i );
                assert( state < ( 1ul << num_bits ) );
            }
        } while( state >= capacity );
        return state;
    }

private:
    uint64_t getCipherBits( uint64_t capacity )
    {
        // Minimum of 8 bits
        if( capacity <= 256 )
            return 8;
        // Subtract one since that is the max value we will use
        capacity--;
        uint64_t i = 0;
        while( capacity != 0 )
        {
            i++;
            capacity >>= 1;
        }
        return i;
    }

    __host__ __device__ uint64_t interleaveWithZero( uint64_t x ) const
    {
        x = ( x | ( x << 16 ) ) & 0x0000FFFF0000FFFF;
        x = ( x | ( x << 8 ) ) & 0x00FF00FF00FF00FF;
        x = ( x | ( x << 4 ) ) & 0x0F0F0F0F0F0F0F0F;
        x = ( x | ( x << 2 ) ) & 0x3333333333333333;
        x = ( x | ( x << 1 ) ) & 0x5555555555555555;
        return x;
    }

    template <uint64_t m, int k>
    __host__ __device__ static inline uint64_t swapbits( uint64_t p )
    {
        uint64_t q = ( ( p >> k ) ^ p ) & m;
        return p ^ q ^ ( q << k );
    }

    __host__ __device__ uint64_t reverseBits( uint64_t n ) const
    {
        static const uint64_t m0 = 0x5555555555555555LLU;
        static const uint64_t m1 = 0x0300c0303030c303LLU;
        static const uint64_t m2 = 0x00c0300c03f0003fLLU;
        static const uint64_t m3 = 0x00000ffc00003fffLLU;
        n = ( ( n >> 1 ) & m0 ) | ( n & m0 ) << 1;
        n = swapbits<m1, 4>( n );
        n = swapbits<m2, 8>( n );
        n = swapbits<m3, 20>( n );
        n = ( n >> 34 ) | ( n << 30 );
        return n;
    }

    __host__ __device__ uint64_t permuteBits( uint64_t a ) const
    {
        uint64_t half_bits = num_bits / 2;
        uint64_t upper_half_bits = num_bits - half_bits;
        uint64_t half_mask = ( 1ull << half_bits ) - 1;
        uint64_t lower_half_split = interleaveWithZero( a & half_mask );
        uint64_t upper_half_split = interleaveWithZero( a >> half_bits );

        return lower_half_split | ( reverseBits( upper_half_split ) >> ( 64 - upper_half_bits * 2 ) );
    }

    __host__ __device__ uint64_t doRound( const uint64_t state, const uint64_t round ) const
    {
        uint64_t output = 0;
        // Do sboxes
        uint64_t i;
        for( i = 0; i <= num_bits - 8; i += 8 )
        {
            output |= sbox256( ( state >> i ) & 0xFF ) << i;
        }
        if( i <= num_bits - 4 )
        {
            output |= sbox16( ( state >> i ) & 0xF ) << i;
            i += 4;
        }
        if( i <= num_bits - 2 )
        {
            output |= sbox4( ( state >> i ) & 0x3 ) << i;
        }
        return permuteBits( output ) ^ key[round];
    }

    uint64_t num_bits;
    uint64_t bit_mask;
    uint64_t capacity;
    uint64_t key[num_rounds];


    __host__ __device__ uint8_t sbox4( uint8_t index ) const
    {
        static const uint8_t sbox4[4] = { 0x3, 0x2, 0x0, 0x1 };
        return sbox4[index];
    }
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
using SPNetworkBijectiveShuffle =
    BijectiveFunctionShuffle<SPNetworkBijectiveFunction<12>, ContainerType, RandomGenerator>;