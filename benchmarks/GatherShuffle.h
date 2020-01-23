#pragma once
#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

// Reads from random addresses
// Doesn't actually shuffle, only used for measurement purposes
template <class BijectiveFunction, class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class GatherShuffle : public Shuffle<ContainerType, RandomGenerator>
{
public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        thrust::gather( in_container.begin(), in_container.end(), in_container.begin(),
                        out_container.begin() );
    }

    bool supportsInPlace() const override
    {
        return false;
    }
};