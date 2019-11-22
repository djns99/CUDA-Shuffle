#pragma once
#include "RandomGenerator.h"

template<class ContainerType, class RandomGenerator>
class Shuffle {
public:
	using container_type = ContainerType;

	/**
	 * Performs random shuffle using the specified seed
	 */
	virtual void shuffle(const ContainerType& in_container, ContainerType& out_container, uint64_t seed) = 0;
	/**
	 * Performs random shuffle using 0 as the seed
	 */
	void shuffle(const ContainerType& in_container, ContainerType& out_container) {
		shuffle(in_container, out_container, 0);
	}

	void operator() (const ContainerType& in_container, ContainerType& out_container, uint64_t seed) {
		shuffle(in_container, out_container, seed);
	}
	void operator() (const ContainerType& in_container, ContainerType& out_container) {
		shuffle(in_container, out_container);
	}
};