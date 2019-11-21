#pragma once
#include "RandomGenerator.h"

template<class Container>
class Shuffle {
public:
	virtual void shuffle(Container& nums, RandomGenerator& random_function) = 0;
	void shuffle(Container& nums) {
		shuffle(nums, DefaultRandomGenerator());
	};
	void operator() (Container& nums, RandomGenerator& random_function) {
		shuffle(nums, rand);
	}

	void operator() (Container& nums) {
		shuffle(nums);
	}
};