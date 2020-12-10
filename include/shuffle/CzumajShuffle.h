#pragma once
#include "WyHash.h"
#include "shuffle/BijectiveFunctionShuffle.h"
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

class CzumajBijection
{
public:
    template <class RandomGenerator>
    void init( uint64_t capacity, RandomGenerator& random_function )
    {
        for( auto& part : key )
        {
            part = random_function();
        }
        num_elements = capacity;
    }

    __host__ __device__ uint64_t operator()( uint64_t val ) const
    {
        return permutator( num_elements, val, 0 );
    }

    constexpr static bool isDeterministic()
    {
        return true;
    }

private:
    static __host__ __device__ uint64_t logFloor( uint64_t n )
    {
#ifdef __CUDA_ARCH__
        return 64 - __clzll( n );
#else
        return 64 - __builtin_clzll( n );
#endif
    }

    __host__ __device__ uint64_t randGen( uint64_t i, uint64_t j ) const
    {
        const uint64_t* const key_start = key + ( i % num_keys ) * key_size;
        const uint64_t res1 = WyHash::wyhash64_v4_key2( key_start, i );
        const uint64_t res2 = WyHash::wyhash64_v4_key2( key_start + 2, j );
        return WyHash::wyhash64_v4_key2( key_start + 4, res1 ^ res2 );
    }

    __host__ __device__ double rejectRand( uint64_t i, uint64_t j ) const
    {
        constexpr uint64_t mantissa_bits = 52;
        constexpr uint64_t mantissa_full_mask = ( 1ull << ( mantissa_bits + 1 ) ) - 1ull; // Includes implicit bit
        constexpr uint64_t mantissa_stored_mask = ( 1ull << mantissa_bits ) - 1ull;
        const uint64_t mantissa = randGen( i, j ) & mantissa_full_mask;
        if( mantissa == 0 )
            return 0;
        const uint64_t top_bit = logFloor( mantissa );
        const uint64_t shift = 1 + mantissa_bits - top_bit;
        const uint64_t exponent = 1024 - shift;
        const uint64_t used_mantissa = ( mantissa << shift ) & mantissa_stored_mask;

        uint64_t val = ( exponent << mantissa_bits ) | used_mantissa;
        return *(double*)&val;
    }

    static __host__ __device__ double intPow( double a, int64_t k )
    {
        if( k < 0 )
            return intPow( 1.0 / a, -k );

        // Fast modular exponentiation
        if( k == 0 )
            return 1;
        double res = 1;
        while( k > 1 )
        {
            if( k & 1 )
            {
                res *= a;
            }
            a *= a;
            k >>= 1;
        }
        return a * res;
    }

    __host__ __device__ uint64_t distCalc( uint64_t a, uint64_t n, uint64_t p, uint64_t k ) const
    {
        // nCk(a, k) * nCk(n-a, p-k) / nCk(n, p)
        // Approximate via nCk(n,k) = n^k/k!
        const uint64_t n_k = n - k;
        const uint64_t p_k = p - k;
        const uint64_t n_a = n - a;
        const uint64_t n_p = n - p;
        const uint64_t n_a_p_k = n - a - p_k;

        uint64_t top_terms = 4;
        uint64_t bottom_terms = 5;

        double result = 1.0;
        // TODO Memoize
        // Calculate the result in one pass of O(n)
        for( uint64_t val = 2; val <= n; val++ )
        {
            if( val > 1 )
            {
                int64_t pow_raise = (int64_t)top_terms - (int64_t)bottom_terms;
                result *= intPow( val, pow_raise );
            }

            if( val == a )
                top_terms--;
            if( val == p )
                top_terms--;
            if( val == n_a )
                top_terms--;
            if( val == n_p )
                top_terms--;

            if( val == k )
                bottom_terms--;
            if( val == n_k )
                bottom_terms--;
            if( val == p_k )
                bottom_terms--;
            if( val == n_a_p_k )
                bottom_terms--;
        }

        return result;
    }

    __host__ __device__ uint64_t repartitorReject( uint64_t n, uint64_t p, uint64_t i ) const
    {
        constexpr double PI = 3.14159265358979323846;
        const uint64_t a = n / 2;
        const double alpha = a / (double)n;
        const double u = p * alpha;
        const double v = 2 * alpha * ( 1 - alpha ) * p;
        const double pow_part = pow( 2, ( 1 - 2 * alpha ) * n ) / pow( 2 * alpha, p );
        const double M = 1.2 * sqrt( v / PI ) * pow_part * sqrt( 1 + p / (double)( n - p ) );

        for( uint64_t l = 0; true; l++ )
        {
            const double rand_1 = rejectRand( i, 2 * l - 1 );
            const double rand_2 = rejectRand( i, 2 * l );
            const double x = sqrt( v ) * tan( PI * rand_1 );
            if( abs( x ) < u )
                continue;
            const uint64_t k = x + u + 0.5;
            if( k > p )
                continue;

            const uint64_t H = distCalc( a, n, p, k );
            if( rand_2 < ( ( ( x * x + v ) * H ) / M ) )
                return k;
        }
    }


    __host__ __device__ uint64_t repartitor( uint64_t n, uint64_t p, uint64_t i ) const
    {
        const uint64_t a = n / 2;
        if( p > a )
            return a - repartitor( n, n - p, i );
        if( p > 10 )
            return repartitorReject( n, p, i );
        uint64_t n1 = a;
        uint64_t n2 = n - a;
        for( uint64_t j = 0; p > 0; p--, j++ )
        {
            const uint64_t r = randGen( i, j ) % ( n1 + n2 );
            if( r < n1 )
                n1--;
            else
                n2--;
        }
        return a - n1;
    }

    __host__ __device__ uint64_t splitter( uint64_t n, uint64_t p, uint64_t x, uint64_t i ) const
    {
        if( n == 1 )
            return x;
        const uint64_t a = n / 2;
        const uint64_t u = repartitor( n, p, i );
        if( x < a )
        {
            const uint64_t t = splitter( a, u, x, i + 1 );
            if( t < u )
                return t;
            else
                return p + ( t - u );
        }
        else
        {
            const uint64_t t = splitter( n - a, p - u, x - a, i + a );
            if( t < ( p - u ) )
                return t + u;
            else
                return t + a;
        }
    }

    static __host__ __device__ uint64_t gamma( uint64_t n )
    {
        const uint64_t f_n = 1 + logFloor( n );
        return n * f_n - ( 1 << f_n ) + 1;
    }

    __host__ __device__ uint64_t permutator( uint64_t n, uint64_t x, uint64_t i ) const
    {
        if( n == 1 )
            return x;
        const uint64_t a = n / 2;
        const uint64_t t = splitter( n, a, x, i );
        if( t < a )
            return permutator( a, t, i + n - 1 );
        else
            return a + permutator( n - a, t - a, i + n - 1 + gamma( a ) );
    }

    constexpr static uint64_t key_size = 6;
    constexpr static uint64_t num_keys = 16;
    uint64_t key[key_size * num_keys];
    uint64_t num_elements;
};


template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using CzumajBijectiveShuffle = BijectiveFunctionShuffle<CzumajBijection, ContainerType, RandomGenerator>;