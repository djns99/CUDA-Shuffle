#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <random>


class DefaultRandomGenerator
{
public:
    typedef uint64_t result_type;

    DefaultRandomGenerator() : random_function( getRandomSeed() )
    {
    }

    DefaultRandomGenerator( uint64_t seed ) : random_function( seed )
    {
    }

    DefaultRandomGenerator( const DefaultRandomGenerator& other ) : random_function( other() )
    {
    }

    DefaultRandomGenerator( DefaultRandomGenerator&& ) = default;

    uint64_t operator()() const
    {
        return random_function();
    }

    constexpr static uint64_t max()
    {
        return std::mt19937_64::max();
    }

    constexpr static uint64_t min()
    {
        return std::mt19937_64::min();
    }

    void discard( uint64_t num )
    {
        random_function.discard( num );
    }

private:
    static uint64_t getRandomSeed()
    {
        uint64_t seed = std::random_device()();
        std::cout << "Using Seed: " << seed << std::endl;
        return seed;
    }

    mutable std::mt19937_64 random_function;
};

class GPURandomGenerator
{
public:
    __device__ GPURandomGenerator(){};
    __device__ GPURandomGenerator( uint64_t seed, uint64_t tid )
    {
        curand_init( seed, tid, 0, &state );
    }

    __device__ uint64_t operator()()
    {
        uint32_t upper = curand( &state );
        uint32_t lower = curand( &state );
        return ( (uint64_t)upper << 32ull ) | (uint64_t)lower;
    }

    __device__ bool getBool()
    {
        // Count the bits in the generated value to get a bool
        return __popcll( curand( &state ) ) & 1;
    }

private:
    curandState state;
};