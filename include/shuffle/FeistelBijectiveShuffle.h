#pragma once
#include "WyHash.h"
#include "shuffle/BijectiveFunctionCompressor.h"
#include "shuffle/BijectiveFunctionScanShuffle.h"
#include "shuffle/BijectiveFunctionShuffle.h"
#include "shuffle/BijectiveFunctionSortShuffle.h"

template <uint64_t num_rounds>
class FeistelBijectiveFunction
{
private:
    struct RoundState
    {
        uint32_t left;
        uint32_t right;
    };

public:
    template <class RandomGenerator>
    void init( uint64_t capacity, RandomGenerator& random_function )
    {
        uint64_t total_bits = getCipherBits( capacity );
        // Half bits rounded down
        left_side_bits = total_bits / 2;
        left_side_mask = ( 1ull << left_side_bits ) - 1;
        // Half the bits rounded up
        right_side_bits = total_bits - left_side_bits;
        right_side_mask = ( 1ull << right_side_bits ) - 1;

        for( uint64_t i = 0; i < num_rounds; i++ )
        {
            key[i][0] = random_function();
            key[i][1] = random_function();
        }
    }

    uint64_t getMappingRange() const
    {
        return 1ull << ( left_side_bits + right_side_bits );
    }
    __device__ uint64_t operator()( const uint64_t val ) const
    {
        // Extract the right and left sides of the input
        uint32_t left = ( uint32_t )( val >> right_side_bits );
        uint32_t right = ( uint32_t )( val & right_side_mask );
        RoundState state = { left, right };

        for( uint64_t i = 0; i < num_rounds; i++ )
        {
            state = doRound( state, i );
        }

        // Check we have the correct number of bits on each side
        assert( ( state.left >> left_side_bits ) == 0 );
        assert( ( state.right >> right_side_bits ) == 0 );

        // Combine the left and right sides together to get result
        return state.left << right_side_bits | state.right;
    }

private:
    uint64_t getCipherBits( uint64_t capacity )
    {
        uint64_t i = 0;
        while( capacity != 0 )
        {
            i++;
            capacity >>= 1;
        }
        return i;
    }


    // __device__ uint32_t applyKey( uint64_t value, const uint64_t key[3] ) const
    // {
    //     // Hash so value affects more than just the lower bits of the key
    //     value = WyHash::wyhash64_v1( value );
    //     // Initialise u,v,w for random number generator
    //     uint64_t u = value ^ key[0];
    //     // Mix the bits so we aren't affecting the same key bits
    //     value ^= value >> 12;
    //     value ^= value << 25;
    //     value ^= value >> 27;
    //     uint64_t v = value ^ key[1];
    //     value ^= value >> 12;
    //     value ^= value << 25;
    //     value ^= value >> 27;
    //     uint64_t w = value ^ key[2];
    //     // Numerical Recipes recommended random number generator
    //     u = u * 2862933555777941757LL + 7046029254386353087LL;
    //     v ^= v >> 17;
    //     v ^= v << 31;
    //     v ^= v >> 8;
    //     w = 4294957665U * ( w & 0xffffffff ) + ( w >> 32 );
    //     uint64_t x = u ^ ( u << 21 );
    //     x ^= x >> 35;
    //     x ^= x << 4;
    //     return ( ( x + v ) ^ w ) & left_side_mask;
    // }

    __device__ uint32_t applyKey( uint64_t value, const uint64_t key[2] ) const
    {
        // Hash so value affects more than just the lower bits of the key
        return WyHash::wyhash64_v3_key2( key, value ) & left_side_mask;
    }

    // __device__ uint32_t applyKey( uint64_t value, const uint64_t key ) const
    // {
    //     // Hash so value affects more than just the lower bits of the key
    //     return WyHash::wyhash64_v3_pair( key, value ) & left_side_mask;
    // }

    __device__ RoundState doRound( const RoundState state, const uint64_t round ) const
    {
        const uint32_t new_left = state.right & left_side_mask;
        const uint32_t round_function_res = state.left ^ applyKey( state.right, key[round] );
        if( right_side_bits != left_side_bits )
        {
            // Upper bit of the old right becomes lower bit of new right if we have odd length feistel
            const uint32_t new_right = ( round_function_res << 1ull ) | state.right >> left_side_bits;
            return { new_left, new_right };
        }
        return { new_left, round_function_res };
    }

    uint64_t right_side_bits;
    uint64_t left_side_bits;
    uint64_t right_side_mask;
    uint64_t left_side_mask;
    uint64_t key[num_rounds][2];
};

static constexpr uint64_t FEISTEL_DEFAULT_NUM_ROUNDS = 8;
template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using FeistelBijectiveShuffle =
    BijectiveFunctionShuffle<BijectiveFunctionCompressor<FeistelBijectiveFunction<FEISTEL_DEFAULT_NUM_ROUNDS>>, ContainerType, RandomGenerator>;

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using FeistelBijectiveSortShuffle =
    BijectiveFunctionSortShuffle<FeistelBijectiveFunction<FEISTEL_DEFAULT_NUM_ROUNDS>, ContainerType, RandomGenerator>;

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using FeistelBijectiveScanShuffle =
    BijectiveFunctionScanShuffle<FeistelBijectiveFunction<FEISTEL_DEFAULT_NUM_ROUNDS>, ContainerType, RandomGenerator>;