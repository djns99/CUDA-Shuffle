#pragma once
#include "Primes.h"
#include "shuffle/BijectiveFunctionShuffle.h"
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

class PeriodicLCGBijection
{
public:
    template <class RandomGenerator>
    void init( uint64_t capacity, RandomGenerator& random_function )
    {
        m = capacity + 1;
        if( m > MAX_CACHED_PRIME * 2 )
            throw std::invalid_argument( "Too large for LCG shuffle" );

        auto it = multiplier_map.find( m );
        if( it == multiplier_map.end() )
        {
            uint64_t a = 1;
            for( auto prime : PRIME_CACHE )
                if( ( m % prime ) == 0 )
                    a *= prime;
            it = multiplier_map.emplace( m, a % m ).first;
        }
        a = it->second;

        // Find a c that is coprime
        uint64_t found_c = 0;
        for( uint64_t i = 1; i < m; i++ )
        {
            if( coprime( i, m ) && ( random_function() % ++found_c ) == 0 )
                c = i;
        }

        // Choose a random start in the sequence
        x0 = random_function() % capacity + 1;
    }

    static uint64_t gcd( uint64_t a, uint64_t b )
    {
        return a == 0 ? b : b == 0 ? a : gcd( a, b % a );
    }

    static uint64_t coprime( uint64_t c, uint64_t m )
    {
        return gcd( c, m ) == 1;
    }

    uint64_t getMappingRange() const
    {
        return m - 1;
    }

    static __host__ __device__ uint64_t powMod( uint64_t a, uint64_t pow, uint64_t m )
    {
        // Fast modular exponentiation
        if( pow == 0 )
            return 1;
        uint64_t res = 1;
        while( pow > 1 )
        {
            if( pow & 1 )
            {
                res *= a;
                res %= m;
            }
            a *= a;
            a %= m;
            pow >>= 1;
        }
        return ( a * res ) % m;
    }

    __host__ __device__ uint64_t operator()( uint64_t val ) const
    {
        val++;
        const uint64_t a_pow = powMod( a, val, m );
        const uint64_t mul = ( a_pow * x0 ) % m;
        const uint64_t add = ( ( ( a_pow - 1 ) * c ) / ( a - 1 ) ) % m;
        return (( mul + add ) % m) - 1;
    }

    constexpr static bool isDeterministic()
    {
        return true;
    }

private:
    uint64_t a;
    uint64_t x0;
    uint64_t c;
    uint64_t m;
    static std::unordered_map<uint64_t, uint64_t> multiplier_map;
};
std::unordered_map<uint64_t, uint64_t> PeriodicLCGBijection::multiplier_map;


template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using PeriodicLCGBijectiveShuffle =
    BijectiveFunctionShuffle<PeriodicLCGBijection, ContainerType, RandomGenerator>;