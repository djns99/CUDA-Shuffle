#include <thrust/device_vector.h>
#include <iostream>
#include <numeric>

#include "shuffle/PrimeFieldBijectiveShuffle.h"
#include "RandomGenerator.h"


int main(void)
{
	PrimeFieldBijectiveShuffle<> shuffle;
	DefaultRandomGenerator gen;
	for(int i = 0; i < 100; i++)
	{
		std::vector<uint64_t> h_nums(30);
		std::iota(h_nums.begin(), h_nums.end(), 0);

		thrust::device_vector<uint64_t> input(h_nums.begin(), h_nums.end());
		thrust::device_vector<uint64_t> output(30, 0);
		shuffle(input, output, gen());

		for(uint64_t num : output)
			std::cout << num << " ";
		std::cout << std::endl;

	}

	int i;
	std::cin >> i;

    return 0;
}