#pragma once
#include <cstdint>
#include <cuda_runtime.h>

class ConstantGenerator
{
public:
    typedef uint64_t result_type;

    __host__ ConstantGenerator() : constant( 0 )
    {
    }

    __host__ ConstantGenerator( uint64_t constant ) : constant( constant )
    {
    }

    __device__ ConstantGenerator( uint64_t constant, uint64_t stride )
        : constant( constant + stride )
    {
    }

    __host__ __device__ uint64_t operator()()
    {
        return constant;
    }

    __host__ __device__ constexpr static uint64_t max()
    {
        return UINT64_MAX;
    }

    __host__ __device__ constexpr static uint64_t min()
    {
        return 0;
    }

private:
    const uint64_t constant;
};