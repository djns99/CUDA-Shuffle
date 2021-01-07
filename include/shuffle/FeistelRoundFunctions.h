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

template <uint64_t num_rounds>
struct CRC32CRoundFunction
{
    template <class RandomGenerator>
    void init( RandomGenerator& gen, uint64_t, uint64_t )
    {
        // NOTE std:: distribution is not broken
        std::uniform_int_distribution<uint64_t> dist;
        for( uint64_t i = 0; i < num_rounds; i++ )
        {
            key[i][0] = dist( gen );
            key[i][1] = dist( gen );
        }
    }

    __host__ __device__ static constexpr uint32_t crc32c( uint64_t input, uint32_t crc = UINT32_MAX )
    {
        // Source: https://en.wikipedia.org/wiki/Computation_of_cyclic_redundancy_checks
        // Uses CRC32C polynomial
#pragma unroll
        for( uint64_t i = 0; i < 64; i += 8 )
        {
            crc ^= ( input >> 8 ) & 0xFF;
#pragma unroll
            for( uint64_t j = 0; j < 8; j++ )
                crc = ( crc >> 1 ) ^ ( 0x82F63B78U * ( crc & 0x1 ) );
        }
        return crc;
    }

    __host__ __device__ static constexpr size_t hashCombine( uint64_t lhs, uint64_t rhs )
    {
        lhs ^= rhs + 0x9e3779b9 + ( lhs << 6 ) + ( lhs >> 2 );
        return lhs;
    }

    __host__ __device__ uint64_t operator()( uint64_t value, uint64_t round ) const
    {
        // TODO Abysmal quality...
        return (uint64_t)crc32c( hashCombine( value, key[round][0] ) ) |
               ( (uint64_t)crc32c( hashCombine( value, key[round][1] ) ) << 32ull );
    }

    uint64_t key[num_rounds][2];
};

template <class DRBG, uint64_t discard, uint64_t num_rounds>
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

    // TODO 64 bit combine?
    __host__ __device__ size_t hashCombine( uint64_t lhs, uint64_t rhs ) const
    {
        lhs ^= rhs + 0x9e3779b9 + ( lhs << 6 ) + ( lhs >> 2 );
        return lhs;
    }

    __host__ __device__ uint64_t operator()( uint64_t value, uint64_t round ) const
    {
        assert( out_bits < 52 && "thrust::uniform_int_distribution requires the number to be "
                                 "representable by a double" );
        thrust::uniform_int_distribution<uint64_t> dist( 0, ( 1ull << out_bits ) - 1 );
        DRBG drbg{ hashCombine( key[round], value ) };
        drbg.discard( discard );
        return dist( drbg );
    }

    uint64_t key[num_rounds];
    uint64_t out_bits;
};

template <uint64_t num_rounds>
using Taus88RoundFunction = DRBGGenerator<thrust::taus88, 0, num_rounds>;

template <uint64_t num_rounds>
using LCGRoundFunction = DRBGGenerator<thrust::default_random_engine, 0, num_rounds>;

template <uint64_t num_rounds>
using Ranlux24RoundFunction = DRBGGenerator<thrust::ranlux24, 0, num_rounds>;

template <uint64_t num_rounds>
using Ranlux48RoundFunction = DRBGGenerator<thrust::ranlux48, 0, num_rounds>;

template <uint64_t num_rounds, class DRBG1, uint64_t discard1, class DRBG2, uint64_t discard2>
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
    __host__ __device__ size_t hashCombine( uint64_t lhs, uint64_t rhs ) const
    {
        lhs ^= rhs + 0x9e3779b9 + ( lhs << 6 ) + ( lhs >> 2 );
        return lhs;
    }

    __host__ __device__ uint64_t operator()( uint64_t value, uint64_t round ) const
    {
        // Combine the key with the value for the seed
        // Otherwise we are fully relying on the randomness of hash combine
        // Note this also ensures Taus88 doesnt get seed of 0
        uint64_t seed = hashCombine( key[round], value );
        DRBG1 drbg1{ seed };
        // Invert seed so we can use the same generator twice
        DRBG2 drbg2{ ~seed };
        drbg1.discard( discard1 );
        drbg2.discard( discard2 );

        // TODO This int distribution is slow. Performance with this removed is 3 GiBs with memory limit of 6 GiBs
        // TODO The 64 bit version is broken hence the need for 2 32 bit invocations
        thrust::uniform_int_distribution<uint32_t> dist;
        uint64_t val1 = (uint64_t)dist( drbg1 ) | ( (uint64_t)dist( drbg1 ) << 32 );
        uint64_t val2 = (uint64_t)dist( drbg2 ) | ( (uint64_t)dist( drbg1 ) << 32 );

        const uint64_t result = hashCombine( val1, val2 );
        return result;
    }

    uint64_t key[num_rounds];
};

template <uint64_t num_rounds>
using Taus88RanluxRoundFunction = DRBGCombiner<num_rounds, thrust::taus88, 1, thrust::ranlux48, 1>;
template <uint64_t num_rounds>
using Taus88LCGRoundFunction =
    DRBGCombiner<num_rounds, thrust::taus88, 1, thrust::default_random_engine, 1>;
template <uint64_t num_rounds>
using RanluxLCGRoundFunction =
    DRBGCombiner<num_rounds, thrust::ranlux48, 1, thrust::default_random_engine, 1>;

template <uint64_t num_rounds>
using DualTaus88RoundFunction = DRBGCombiner<num_rounds, thrust::taus88, 1, thrust::taus88, 1>;
template <uint64_t num_rounds>
using DualLCGRoundFunction =
    DRBGCombiner<num_rounds, thrust::default_random_engine, 1, thrust::default_random_engine, 1>;
template <uint64_t num_rounds>
using DualRanluxRoundFunction = DRBGCombiner<num_rounds, thrust::ranlux48, 1, thrust::ranlux48, 1>;
