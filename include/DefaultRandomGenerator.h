#pragma once
#include <random>
#include <curand_kernel.h>

class DefaultRandomGenerator
{
public:
    using result_type = uint64_t;

    DefaultRandomGenerator() : random_function( std::random_device()() )
    {
    }

    DefaultRandomGenerator( uint64_t seed ) : random_function( seed )
    {
    }

    DefaultRandomGenerator( DefaultRandomGenerator& other )
        : random_function( other() )
    {
    }

    uint64_t operator()()
    {
        return random_function();
    }

    uint64_t max() {
        return UINT64_MAX;
    }

    uint64_t min() {
        return 0;
    }
private:
    // thrust::xor_combine_engine<thrust::linear_congruential_engine<uint64_t, 6364136223846793005U, 1442695040888963407U, 0U>, 0, thrust::ranlux48_base, 0> random_function;
    std::mt19937_64 random_function;
};

class GPURandomGenerator
{
public:
    __device__ GPURandomGenerator( uint64_t seed, uint64_t tid )
    {
        curand_init( seed, tid, 0, &state );
    }

    __device__ uint64_t operator()()
    {
        uint32_t upper = curand(&state);
        uint32_t lower = curand(&state);
        return ((uint64_t)upper << 32ull) | (uint64_t)lower;
    }

    __device__ bool getBool() {
        // Count the bits in the generated value to get a bool
        return __popcll(curand(&state)) & 1;
    }

private:
    curandState state;
};