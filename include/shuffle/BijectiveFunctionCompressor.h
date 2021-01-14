#pragma once
#include <assert.h>

template <class BijectiveFunction>
class BijectiveFunctionCompressor : BijectiveFunction
{
public:
    template <class RandomGenerator>
    void init( uint64_t capacity, RandomGenerator& random_function )
    {
        this->capacity = capacity;
        BijectiveFunction::init( capacity, random_function );
    }

    __host__ __device__ uint64_t operator()( uint64_t val ) const
    {

        do
        {
            const uint64_t last_val = val;
            (void)last_val;
            val = BijectiveFunction::operator()( val );
            assert( val != last_val || val < capacity );
        } while( val >= capacity );
        return val;
    }

private:
    uint64_t capacity;
};