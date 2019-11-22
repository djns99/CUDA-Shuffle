#pragma once
#include <vector>

#include "shuffle/Shuffle.h"

template<class ContainerType = std::vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class FisherYatesShuffle : public Shuffle<ContainerType, RandomGenerator>
{
private:
	void swap(ContainerType& container, uint64_t a, uint64_t b)
	{
		auto temp = container[a];
		container[a] = container[b];
		container[b] = temp;
	}
public:
	void shuffle(const ContainerType& in_container, ContainerType& out_container, uint64_t seed) override
	{
		if (&in_container != &out_container)
		{
			// Copy if we are not doing an inplace operation
			std::copy(in_container.begin(), in_container.end(), out_container.begin());
		}
		
		RandomGenerator random_function(seed);
		const uint64_t capacity = out_container.size();
		for (uint64_t i = 0; i < capacity - 1; i++)
		{
			auto swap_range = capacity - i;
			swap(out_container, i, i + (random_function() % swap_range));
		}
	}
};
