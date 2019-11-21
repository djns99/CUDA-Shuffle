#pragma once
#include <vector>

#include "shuffle/Shuffle.h"

template<Container = std::vector<uint64_t>>
class FisherYatesShuffle : public Shuffle<Container>
{
private:
	void swap(Container& nums, uint64_t a, uint64_t b)
	{
		auto temp = nums[a];
		nums[a] = nums[b];
		nums[b] = temp;
	}
public:
	void shuffle(Container& nums, RandomGenerator& random_function) override
	{
		const uint64_t capacity = nums.size();
		for (uint64_t i = 0; i < capacity - 1; i++)
		{
			auto swap_range = capacity - i;
			swap(nums, i, i + (random_function() % swap_range));
		}
	}
};
