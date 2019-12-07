#pragma once
#include "shuffle/BijectiveFunctionShuffle.h"

class PrimeFieldBijectiveFunction
{
public:
    PrimeFieldBijectiveFunction()
    {
    }

    template <class RandomGenerator>
    void init( uint64_t capacity, RandomGenerator& random_function )
    {
        this->capacity = capacity;
        modulus = roundUpPower2( capacity );
        // Must be odd so it is coprime to modulus
        multiplier = ( random_function() * 2 + 1 ) % modulus;
        pre_addition = random_function() % modulus;
        post_addition = random_function() % modulus;
    }

    __host__ __device__ uint64_t operator()( uint64_t val ) const
    {
        // Modulus must be power of two
        assert( ( modulus & ( modulus - 1 ) ) == 0 );
        do
        {
            val = ( ( ( val + pre_addition ) * multiplier ) + post_addition ) & ( modulus - 1 );
        } while( val >= capacity );
        return val;
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

    uint64_t capacity;
    uint64_t modulus;
    uint64_t multiplier;
    uint64_t pre_addition;
    uint64_t post_addition;
};

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using PrimeFieldBijectiveShuffle =
    BijectiveFunctionShuffle<PrimeFieldBijectiveFunction, ContainerType, RandomGenerator>;