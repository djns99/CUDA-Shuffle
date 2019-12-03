#pragma once

class ConstantGenerator
{
public:
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

private:
    uint64_t constant;
};