#pragma once
#include <vector>

#include "shuffle/Shuffle.h"

template<class ContainerType = std::vector<uint64_t>>
class FisherYatesShuffle : public Shuffle<ContainerType>
{
private:
	void swap(ContainerType& container, uint64_t a, uint64_t b)
	{
		auto temp = container[a];
		container[a] = container[b];
		container[b] = temp;
	}
public:
	void shuffle(ContainerType& container, RandomGenerator& random_function) override
	{
		const uint64_t capacity = container.size();
		for (uint64_t i = 0; i < capacity - 1; i++)
		{
			auto swap_range = capacity - i;
			swap(container, i, i + (random_function() % swap_range));
		}
	}
};
