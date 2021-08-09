#pragma once
#include "DefaultRandomGenerator.h"
#include "ThrustInclude.h"
#include "shuffle/Shuffle.h"
#include <vector>

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class ThrustShuffle : public Shuffle<ContainerType, RandomGenerator>
{
public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        RandomGenerator g( seed );
        thrust::shuffle_copy( in_container.begin(), in_container.begin() + num, out_container.begin(), g );
    }

    bool supportsInPlace() const override
    {
        return false;
    }
};