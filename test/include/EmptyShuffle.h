#pragma once
#include "ThrustInclude.h"
#include "shuffle/Shuffle.h"

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class EmptyShuffle : public Shuffle<ContainerType, RandomGenerator>
{
public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        if( &in_container != &out_container )
        {
            // Copy if we are not doing an inplace operation
            thrust::copy( in_container.begin(), in_container.begin() + num, out_container.begin() );
        }
    }
};