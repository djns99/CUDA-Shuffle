/**
 * WyHash implementation based of implementation https://github.com/wangyi-fudan/wyhash
 */
#pragma once
#include <cuda_runtime.h>
#include <cassert>

class WyHash
{
private:
    constexpr static uint64_t _wyp0 = 0xa0761d6478bd642full;
    constexpr static uint64_t _wyp1 = 0xe7037ed1a0b428dbull;
    constexpr static uint64_t _wyp2 = 0x8ebc6af09c88c6e3ull;
    constexpr static uint64_t _wyp3 = 0x589965cc75374cc3ull;
    constexpr static uint64_t _wyp4 = 0x1d8e4e27c47d124full;

    __host__ __device__ uint64_t rotr( uint64_t val, uint64_t i )
    {
        return ( val >> i ) | ( val << ( 64 - i ) );
    }

    __host__ __device__ static uint64_t mulhi( uint64_t a, uint64_t b )
    {


#ifdef __CUDA_ARCH__
        return __umul64hi( a, b );
#else
        uint64_t a1 = a & UINT32_MAX;
        uint64_t a2 = a >> 32;
        uint64_t b1 = b & UINT32_MAX;
        uint64_t b2 = b >> 32;

        uint64_t a1_b1 = a1 * b1;
        uint64_t a2_b1 = a2 * b1 + ( a1_b1 >> 32 );
        uint64_t a1_b2 = a1 * b2;
        uint64_t a2_b2 = a2 * b2 + ( a1_b2 >> 32 );
        uint64_t sum2 = ( a2_b1 & UINT32_MAX ) + ( a1_b2 & UINT32_MAX );
        uint64_t sum34 = a2_b2 + ( sum2 >> 32 ) + ( a2_b1 >> 32 );
        return sum34;
#endif
    }

    __host__ __device__ static uint64_t mum( uint64_t a, uint64_t b )
    {
        return mulhi( a, b ) ^ ( a * b );
    }

public:
    __host__ __device__ static uint64_t wyhash64_v4_key2( const uint64_t key[2], uint64_t seed )
    {
        return mum( mum( key[0] ^ seed ^ _wyp0, key[1] ^ seed ^ _wyp1 ) ^ seed, 16 ^ _wyp4 );
    }

    __host__ __device__ static uint64_t wyhash64_v3_key2( const uint64_t key[2], uint64_t seed )
    {
        return mum( mum( key[0] ^ seed ^ _wyp0, key[1] ^ seed ^ _wyp1 ), 16 ^ _wyp4 );
    }

    __host__ __device__ static uint64_t wyhash64_v3_key4( const uint64_t key[4], uint64_t seed )
    {
        return mum( mum( key[0] ^ seed ^ _wyp0, key[1] ^ seed ^ _wyp1 ) ^
                        mum( key[2] ^ seed ^ _wyp2, key[3] ^ seed ^ _wyp4 ),
                    32 ^ _wyp4 );
    }

    __host__ __device__ static uint64_t wyhash64_v3_pair( uint64_t a, uint64_t b )
    {
        return mum( mum( a ^ _wyp0, b ^ _wyp1 ), _wyp2 );
    }

    /*
     * wyhash64 hash function version 1
     */
    __host__ __device__ static uint64_t wyhash64_v1( uint64_t wyhash64 )
    {
        wyhash64 += 0x60bee2bee120fc15;
        uint64_t x = wyhash64;
        constexpr uint64_t y = 0xa3b195354a39b70d;
        constexpr uint64_t z = 0x1b03738712fad5c9;
        uint64_t w = mulhi( x, y ) ^ ( x * y );
        return mulhi( z, w ) ^ ( z * w );
    }
};