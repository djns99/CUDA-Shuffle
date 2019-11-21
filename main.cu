#include <thrust/device_vector.h>
#include <iostream>
#include <numeric>

#include "shuffle/PrimeFieldShuffle.h"
#include "RandomGenerator.h"


int main(void)
{
	PrimeFieldShuffle<> shuffle;
	DefaultRandomGenerator gen;
	for(int i = 0; i < 10; i++)
	{
		std::vector<uint64_t> h_nums(10);
		std::iota(h_nums.begin(), h_nums.end(), 0);

		thrust::device_vector<uint64_t> nums(h_nums.begin(), h_nums.end());
		shuffle.shuffle(nums, gen);

		for(uint64_t num : nums)
			std::cout << num << " ";
		std::cout << std::endl;

	}

	int i;
	std::cin >> i;

    return 0;
}