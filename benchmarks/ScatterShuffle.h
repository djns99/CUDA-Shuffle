#pragma once
#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"
#include <thrust/device_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/scatter.h>

// Writes into random addresses
// Doesn't actually shuffle, only used for measurement purposes
template <class BijectiveFunction, class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class ScatterShuffle : public Shuffle<ContainerType, RandomGenerator>
{
public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        thrust::counting_iterator<uint64_t> counting( 0 );
        auto scatter_key = thrust::make_transform_iterator( counting, [=] __device__( uint64_t idx ) {
            thrust::minstd_rand rng( seed );
            rng.discard( idx );

            thrust::uniform_int_distribution<uint64_t> dist( 0, num - 1 );
            return dist( rng );
        } );
        thrust::scatter( in_container.begin(), in_container.end(), scatter_key, out_container.begin() );
    }

    bool supportsInPlace() override
    {
        return false;
    }
};