#pragma once
#include "DefaultRandomGenerator.h"
#include "mergeshuffle/MergeShuffle.h"
#include "shuffle/Shuffle.h"
#include <vector>

template <class ContainerType = std::vector<uint32_t>, class RandomGenerator = DefaultRandomGenerator>
class MergeShuffle : public Shuffle<ContainerType, RandomGenerator>
{
public:
    // TODO Investigate why this & rao is broken
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        if( &in_container != &out_container )
        {
            // Copy if we are not doing an inplace operation
            std::copy( in_container.begin(), in_container.begin() + num, out_container.begin() );
        }

        RandomGenerator g( seed );
        setMergeShuffleRand64( [&g]() { return g(); } );
        parallel_merge_shuffle( out_container.data(), num );
        setMergeShuffleRand64( {} );
    }

    bool isDeterministic() const override
    {
        return false;
    }
};