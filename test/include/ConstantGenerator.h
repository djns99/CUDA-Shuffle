#pragma once
#include <cstdint>

class ConstantGenerator
{
public:
    using result_type = uint64_t;

    ConstantGenerator() : constant( 0 )
    {
    }

    ConstantGenerator( uint64_t constant ) : constant( constant )
    {
    }

    __host__ __device__ uint64_t operator()()
    {
        return constant;
    }

    __host__ __device__ uint64_t max() {
        return UINT64_MAX;
    }

    __host__ __device__ uint64_t min() {
        return 0;
    }

private:
    uint64_t constant;
};