#pragma once

#include "ThrustInclude.h"
#include "WyHash.h"
#include <random>

template <uint64_t num_rounds>
struct WyHashRoundFunction
{
    template <class RandomGenerator>
    void init( RandomGenerator& gen, uint64_t, uint64_t )
    {
        std::uniform_int_distribution<uint64_t> dist;
        for( uint64_t i = 0; i < num_rounds; i++ )
        {
            key[i][0] = dist( gen );
            key[i][1] = dist( gen );
        }
    }

    __host__ __device__ uint64_t operator()( uint64_t value, uint64_t round ) const
    {
        return WyHash::wyhash64_v4_key2( key[round], value );
    }

    uint64_t key[num_rounds][2];
};

template <class DRBG, uint64_t num_rounds>
struct DRBGGenerator
{
    template <class RandomGenerator>
    void init( RandomGenerator& gen, uint64_t, uint64_t _out_bits )
    {
        std::uniform_int_distribution<uint64_t> dist;
        for( uint64_t i = 0; i < num_rounds; i++ )
        {
            key[i] = dist( gen );
        }

        out_bits = _out_bits;
    }

    __host__ __device__ uint64_t operator()( uint64_t value, uint64_t round ) const
    {
        thrust::uniform_int_distribution<uint64_t> dist( 0, ( 1ull << out_bits ) - 1 );
        DRBG t{ value ^ key[round] };
        return dist( t );
    }

    uint64_t key[num_rounds];
    uint64_t out_bits;
};

template <uint64_t num_rounds>
using Taus88RoundFunction = DRBGGenerator<thrust::taus88, num_rounds>;

template <uint64_t num_rounds>
using LCGRoundFunction = DRBGGenerator<thrust::default_random_engine, num_rounds>;

template <uint64_t num_rounds>
using Ranlux24RoundFunction = DRBGGenerator<thrust::ranlux24, num_rounds>;

template <uint64_t num_rounds>
using Ranlux48RoundFunction = DRBGGenerator<thrust::ranlux48, num_rounds>;

template <uint64_t num_rounds, class DRBG1, class DRBG2>
struct DRBGCombiner
{
    template <class RandomGenerator>
    void init( RandomGenerator& gen, uint64_t, uint64_t )
    {
        std::uniform_int_distribution<uint64_t> dist;
        for( uint64_t i = 0; i < num_rounds; i++ )
        {
            key[i] = dist( gen );
        }
    }

    // TODO 64 bit combine?
    __host__ __device__ size_t hash_combine( uint64_t lhs, uint64_t rhs ) const
    {
        lhs ^= rhs + 0x9e3779b9 + ( lhs << 6 ) + ( lhs >> 2 );
        return lhs;
    }

    __host__ __device__ uint64_t operator()( uint64_t value, uint64_t round ) const
    {
        // Combine the key with the value for the seed
        // Otherwise we are fully relying on the randomness of hash combine
        uint64_t seed = hash_combine( value, key[round] );
        DRBG1 drbg1{ seed };
        DRBG2 drbg2{ seed };
        thrust::uniform_int_distribution<uint64_t> dist;
        return hash_combine( dist( drbg1 ), dist( drbg2 ) );
    }

    uint64_t key[num_rounds];
};

template<uint64_t num_rounds>
using Taus88RanluxRoundFunction = DRBGCombiner<num_rounds, thrust::taus88, thrust::ranlux48>;
template<uint64_t num_rounds>
using Taus88LCGRoundFunction = DRBGCombiner<num_rounds, thrust::taus88, thrust::default_random_engine>;
template<uint64_t num_rounds>
using RanluxLCGRoundFunction = DRBGCombiner<num_rounds, thrust::ranlux48, thrust::default_random_engine>;