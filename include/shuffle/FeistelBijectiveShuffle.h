#pragma once
#include "shuffle/BijectiveFunctionCompressor.h"
#include "shuffle/BijectiveFunctionScanShuffle.h"
#include "shuffle/BijectiveFunctionShuffle.h"
#include "shuffle/BijectiveFunctionSortShuffle.h"
#include "shuffle/FeistelRoundFunctions.h"

template <uint64_t num_rounds, class RoundFunction = WyHashRoundFunction<num_rounds>>
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

        round_function.init( random_function, right_side_bits, left_side_bits );
    }

    uint64_t getMappingRange() const
    {
        return 1ull << ( left_side_bits + right_side_bits );
    }

    __host__ __device__ uint64_t operator()( const uint64_t val ) const
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
        return (uint64_t)state.left << right_side_bits | (uint64_t)state.right;
    }

    constexpr static bool isDeterministic()
    {
        return true;
    }

private:
    uint64_t getCipherBits( uint64_t capacity )
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

        if( std::is_same_v< RC5RoundFunction<num_rounds>, RoundFunction> )
            return std::max( i, uint64_t( 8 ) ); // Minimum number required for good results
        else
            return i;
    }

    __host__ __device__ uint32_t applyRoundFunction( RoundState state, uint64_t round ) const
    {
        // Hash so value affects more than just the lower bits of the key
        return state.left ^ (round_function( state.right, round, state.left ) & left_side_mask);
    }

    // __host__ __device__ uint32_t applyRoundFunction( uint64_t value, const uint64_t key ) const
    // {
    //     // Hash so value affects more than just the lower bits of the key
    //     return WyHash::wyhash64_v3_pair( key, value ) & left_side_mask;
    // }

    __host__ __device__ RoundState doRound( const RoundState state, const uint64_t round ) const
    {
        assert( state.right <= right_side_mask);
        assert( state.left <= left_side_mask);
        const uint32_t new_left = state.right & left_side_mask;
        const uint32_t round_function_res = applyRoundFunction( state, round );
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
    RoundFunction round_function;
};

static constexpr uint64_t FEISTEL_DEFAULT_NUM_ROUNDS = 16;
template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using FeistelBijectiveShuffle =
    BijectiveFunctionShuffle<BijectiveFunctionCompressor<FeistelBijectiveFunction<FEISTEL_DEFAULT_NUM_ROUNDS>>, ContainerType, RandomGenerator>;

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using FeistelBijectiveSortShuffle =
    BijectiveFunctionSortShuffle<FeistelBijectiveFunction<FEISTEL_DEFAULT_NUM_ROUNDS>, ContainerType, RandomGenerator>;

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using FeistelBijectiveScanShuffle =
    BijectiveFunctionScanShuffle<FeistelBijectiveFunction<FEISTEL_DEFAULT_NUM_ROUNDS>, ContainerType, RandomGenerator>;