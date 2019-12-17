#pragma once
#include "shuffle/BijectiveFunctionCompressor.h"
#include "shuffle/BijectiveFunctionScanShuffle.h"
#include "shuffle/BijectiveFunctionShuffle.h"
#include "shuffle/BijectiveFunctionSortShuffle.h"

class PrimeFieldBijectiveFunction
{
public:
    PrimeFieldBijectiveFunction()
    {
    }

    template <class RandomGenerator>
    void init( uint64_t capacity, RandomGenerator& random_function )
    {
        modulus = roundUpPower2( capacity );
        // Must be odd so it is coprime to modulus
        multiplier = ( random_function() * 2 + 1 ) % modulus;
        pre_addition = random_function() % modulus;
        post_addition = random_function() % modulus;
    }

    uint64_t getMappingRange() const
    {
        return modulus;
    }

    __host__ __device__ uint64_t operator()( uint64_t val ) const
    {
        // Modulus must be power of two
        assert( ( modulus & ( modulus - 1 ) ) == 0 );
        return ( ( ( val + pre_addition ) * multiplier ) + post_addition ) & ( modulus - 1 );
    }

private:
    static uint64_t roundUpPower2( uint64_t a )
    {
        if( a & ( a - 1 ) )
        {
            uint64_t i;
            for( i = 0; a > 1; i++ )
            {
                a >>= 1ull;
            }
            return 1ull << ( i + 1ull );
        }
        return a;
    }

    uint64_t modulus;
    uint64_t multiplier;
    uint64_t pre_addition;
    uint64_t post_addition;
};

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using PrimeFieldBijectiveShuffle =
    BijectiveFunctionShuffle<BijectiveFunctionCompressor<PrimeFieldBijectiveFunction>, ContainerType, RandomGenerator>;

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using PrimeFieldBijectiveSortShuffle =
    BijectiveFunctionSortShuffle<PrimeFieldBijectiveFunction, ContainerType, RandomGenerator>;

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using PrimeFieldBijectiveScanShuffle =
    BijectiveFunctionScanShuffle<PrimeFieldBijectiveFunction, ContainerType, RandomGenerator>;