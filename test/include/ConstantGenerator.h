#pragma once
#include <cstdint>

class ConstantGenerator
{
public:
    typedef uint64_t result_type;

    ConstantGenerator() : constant( 0 )
    {
    }

    ConstantGenerator( uint64_t constant ) : constant( constant )
    {
    }

    uint64_t operator()()
    {
        return constant;
    }

    constexpr static uint64_t max()
    {
        return UINT64_MAX;
    }

    constexpr static uint64_t min()
    {
        return 0;
    }

private:
    const uint64_t constant;
};