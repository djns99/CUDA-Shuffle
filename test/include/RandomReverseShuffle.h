#pragma once
#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"
#include <vector>

template <class ContainerType = std::vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class RandomReverseShuffle : public Shuffle<ContainerType, RandomGenerator>
{
private:
    void swap( ContainerType& container, uint64_t a, uint64_t b )
    {
        auto temp = container[a];
        container[a] = container[b];
        container[b] = temp;
    }

public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        if( &in_container != &out_container )
        {
            // Copy if we are not doing an inplace operation
            std::copy( in_container.begin(), in_container.begin() + num, out_container.begin() );
        }

        RandomGenerator gen( seed );
        for( uint64_t i = 0; i < num / 2; i++ )
        {
            if( gen() % 2 )
            {
                swap( out_container, i, num - 1 - i );
            }
        }
    }
};