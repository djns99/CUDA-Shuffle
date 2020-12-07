#pragma once
#include "shuffle/BijectiveFunctionScanShuffle.h"
#include "shuffle/BijectiveFunctionShuffle.h"
#include "shuffle/BijectiveFunctionSortShuffle.h"

class NoOpBijectiveFunction
{
public:
    template <class RandomGenerator>
    void init( uint64_t capacity, RandomGenerator& random_function )
    {
        // Round up to a power of two to emulate a zero cost n-bit function
        this->capacity = roundUpPower2( capacity );
    }

    uint64_t getMappingRange() const
    {
        return capacity;
    }

    __host__ __device__ uint64_t operator()( uint64_t val ) const
    {
        return val;
    }

    constexpr static bool isDeterministic()
    {
        return true;
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
};

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using NoOpBijectiveShuffle =
    BijectiveFunctionShuffle<NoOpBijectiveFunction, ContainerType, RandomGenerator>;

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using NoOpBijectiveSortShuffle =
    BijectiveFunctionSortShuffle<NoOpBijectiveFunction, ContainerType, RandomGenerator>;

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using NoOpBijectiveScanShuffle =
    BijectiveFunctionScanShuffle<NoOpBijectiveFunction, ContainerType, RandomGenerator>;