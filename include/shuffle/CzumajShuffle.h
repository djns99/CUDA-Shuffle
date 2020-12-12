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
        return n == 0 ? 0 : 64 - __builtin_clzll( n );
#endif
    }

    __host__ __device__ uint64_t randGen( uint64_t i, uint64_t j ) const
    {
        const uint64_t* const key_start = key + ( i % num_keys ) * key_size;
        assert( key_start < key + num_keys * key_size );

        for( uint64_t round = 2; round < key_size; round += 4)
        {
            i = WyHash::wyhash64_v4_key2( key_start + round, i );
            j = WyHash::wyhash64_v4_key2( key_start + round + 2, j );
        }
        return WyHash::wyhash64_v4_key2( key_start, i ^ j );
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


    static __host__ __device__ double distCalc( uint64_t a, uint64_t n, uint64_t p, uint64_t k )
    {
        // nCr(a, k) * nCr(n-a, p-k) / nCr(n, p)
        const uint64_t a_k = a - k;
        const uint64_t p_k = p - k;
        const uint64_t n_a = n - a;
        const uint64_t n_p = n - p;
        const uint64_t n_a_p_k = n_a - p_k;

        double res = 1.0;
        for( uint64_t val = 2; val <= n; val++ )
        {
            const auto d_val = (double)val;

            // nCr(a, k)
            if( val <= k )
                res /= d_val;
            if( val > a_k && val <= a )
                res *= d_val;

            // nCr(n-a, p-k)
            if( val <= p_k )
                res /= d_val;
            if( val > n_a_p_k && val <= n_a )
                res *= d_val;

            // nCr(n, p)
            if( val <= p )
                res *= d_val;
            if( val > n_p && val <= n )
                res /= d_val;
        }

        return res;
    }

    __host__ __device__ uint64_t repartitorReject( uint64_t n, uint64_t p, uint64_t i ) const
    {
        constexpr double PI = 3.14159265358979323846;
        const uint64_t a = n / 2;
        const double alpha = a / (double)n;
        const double u = p * alpha;
        const double v = 2 * alpha * ( 1 - alpha ) * p;
        const double pow_part = (double)pow( 2.0, ( 1 - 2.0 * alpha ) * n ) / (double)pow( 2.0 * alpha, p );
        const double M = 1.2 * sqrt( v / PI ) * pow_part * sqrt( 1 + p / (double)( n - p ) );

        for( uint64_t l = 1; true; l++ )
        {
            const double rand_1 = rejectRand( i, 2 * l - 1 );
            const double x = sqrt( v ) * tan( PI * rand_1 );
            const double k = floor( x + u + 0.5 );
            if( k < 0 || k > p )
                continue;

            const double H = distCalc( a, n, p, k );
            const double rand_2 = rejectRand( i, 2 * l );
            const double comp = ( x * x + v ) * ( H / M );
            if( rand_2 <= comp )
                return k;
        }
    }


    __host__ __device__ uint64_t repartitor( uint64_t n, uint64_t p, uint64_t i ) const
    {
        const uint64_t a = n / 2;
        bool switch_p = p > a;
        if( switch_p )
            p = n - p;
        uint64_t res;
        if( p <= 10 )
        {
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
            res = a - n1;
        }
        else
        {
            res = repartitorReject( n, p, i );
        }

        if( switch_p )
            return a - res;
        return res;
    }

    /*__host__ __device__ uint64_t splitter_r( uint64_t n, uint64_t p, uint64_t x, uint64_t i ) const
    {
        assert( n <= num_elements );
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

        // Should never get here
        assert( false );
    }*/

    __host__ __device__ uint64_t splitter( uint64_t n, uint64_t p, uint64_t x, uint64_t i ) const
    {
        constexpr uint64_t max_bits = 40;
        assert( n >= 1 );
        assert( n < ( 1ull << max_bits ) );

        uint64_t p_stack[max_bits];
        uint64_t u_stack[max_bits];
        uint64_t a_stack[max_bits];
        bool dir_stack[max_bits];

        int64_t depth;
        for( depth = 0; n != 1; depth++ )
        {
            const uint64_t a = n / 2;
            const uint64_t u = repartitor( n, p, i );
            a_stack[depth] = a;
            u_stack[depth] = u;
            p_stack[depth] = p;
            dir_stack[depth] = x < a;
            if( x < a )
            {
                n = a;
                p = u;
                x = x;
                i++;
            }
            else
            {
                n -= a;
                p -= u;
                x -= a;
                i += a;
            }
        }

        uint64_t t = x;
        for( depth--; depth >= 0; depth-- )
        {
            if( dir_stack[depth] )
            {
                if( t < u_stack[depth] )
                    t = t;
                else
                    t = p_stack[depth] + ( t - u_stack[depth] );
            }
            else
            {
                if( t < ( p_stack[depth] - u_stack[depth] ) )
                    t += u_stack[depth];
                else
                    t += a_stack[depth];
            }
        }
        return t;
    }


    static __host__ __device__ uint64_t gamma( uint64_t n )
    {
        const uint64_t f_n = 1 + logFloor( n );
        return n * f_n - ( 1 << f_n ) + 1;
    }

    __host__ __device__ uint64_t permutator( uint64_t n, uint64_t x, uint64_t i ) const
    {
        assert( n >= 1 );
        uint64_t sum = 0;
        for( uint64_t depth = 0; n != 1; depth++ )
        {
            uint64_t a = n / 2;
            const uint64_t t = splitter( n, a, x, i );
            if( t < a )
            {
                i = i + n - 1;
                n = a;
                x = t;
            }
            else
            {
                sum += a;
                i = i + n - 1 + gamma( a );
                n = n - a;
                x = t - a;
            }
        }

        assert( ( sum + x ) < num_elements );
        return sum + x;
    }

    constexpr static uint64_t key_size = 14;
    constexpr static uint64_t num_keys = 16;
    uint64_t key[key_size * num_keys];
    uint64_t num_elements;
};


template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using CzumajBijectiveShuffle = BijectiveFunctionShuffle<CzumajBijection, ContainerType, RandomGenerator>;