#include <iostream>
#include <numeric>
#include <iomanip>

#include "shuffle/FisherYatesShuffle.h"


int main(void)
{
	FisherYatesShuffle<> shuffle;
	DefaultRandomGenerator gen;
	for(int i = 0; i < 1; i++)
	{
		std::vector<uint64_t> h_nums(8);
		std::iota(h_nums.begin(), h_nums.end(), 0);
        shuffle( h_nums, h_nums, gen() );

		std::cout << "{";
        for( uint64_t num : h_nums )
			std::cout << num * 8 << ", ";
		std::cout << "}" << std::endl;

	}


    return 0;
}