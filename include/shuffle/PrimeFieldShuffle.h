#pragma once
#include <thrust/device_vector.h>

#include <thrust/sequence.h>

#include "shuffle/Shuffle.h"

template<class Container=thrust::device_vector<uint64_t>>
class PrimeFieldShuffle : public Shuffle<Container>
{
private:
	static uint64_t roundUpPower2(uint64_t a)
	{
		if (a & (a - 1))
		{
			uint64_t i;
			for (i = 0; a > 1; i++) {
				a >>= 1;
			}
			return 1 << (i + 1);
		}
		return a;
	}

public:
	void shuffle(Container& nums, RandomGenerator& random_function) override
	{
		// Round up to power of two
		uint64_t cap = roundUpPower2(nums.size());
		// Choose an odd number so we know it is coprime with cap
		uint64_t mul = (random_function() * 2 + 1) % cap;
		// Choose a shift
		uint64_t shift = random_function() % cap;

		thrust::device_vector<uint64_t> keys(nums.size());

		// Initialise key vector with indexes
		thrust::sequence(keys.begin(), keys.end());
		// Inplace transform
		thrust::transform(keys.begin(), keys.end(), keys.begin(), [=] __host__ __device__(uint64_t val) -> uint64_t {
			return (val * mul + shift) % cap;
		});
		// Sort by keys
		thrust::sort_by_key(keys.begin(), keys.end(), nums.begin());
	}
};