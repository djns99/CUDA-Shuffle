#pragma once

#include "ThrustInclude.h"
#include "WyHash.h"
#include <random>

template <uint64_t num_rounds>
struct WyHashRoundFunction
{
    template <class RandomGenerator>
    void init( RandomGenerator& gen, uint64_t in_bits, uint64_t out_bits )
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

template<class DRBG, uint64_t num_rounds>
struct DRBGGenerator {
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
        thrust::uniform_int_distribution<uint64_t> dist( 0, (1ull << out_bits) - 1);
        DRBG t{ value ^ key[round] };
        return dist( t );
    }

    uint64_t key[num_rounds];
    uint64_t out_bits;
};

template<uint64_t num_rounds>
using Taus88RoundFunction = DRBGGenerator<thrust::taus88, num_rounds>;

template<uint64_t num_rounds>
using LCGRoundFunction = DRBGGenerator<thrust::default_random_engine , num_rounds>;

template<uint64_t num_rounds>
using Ranlux24RoundFunction = DRBGGenerator<thrust::ranlux24, num_rounds>;

template<uint64_t num_rounds>
using Ranlux48RoundFunction = DRBGGenerator<thrust::ranlux48, num_rounds>;
