#pragma once
#include <vector>
#include <algorithm>
#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"

template <class ContainerType = std::vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class StdShuffle : public Shuffle<ContainerType, RandomGenerator>
{
public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        if( &in_container != &out_container )
        {
            // Copy if we are not doing an inplace operation
            std::copy( in_container.begin(), in_container.begin() + num, out_container.begin() );
        }

        RandomGenerator g(seed);
        std::shuffle( out_container.begin(), out_container.begin() + num, g );
    }
};