#pragma once
#include "RandomGenerator.h"

template<class ContainerType>
class Shuffle {
public:
	using container_type = ContainerType;

	virtual void shuffle(ContainerType& container, RandomGenerator& random_function) = 0;
	void shuffle(ContainerType& container) {
		shuffle(container, DefaultRandomGenerator());
	};
	void operator() (ContainerType& container, RandomGenerator& random_function) {
		shuffle(container, rand);
	}

	void operator() (ContainerType& container) {
		shuffle(container);
	}
};