#pragma once
#include "RandomGenerator.h"

template<class ContainerType, class RandomGenerator>
class Shuffle {
public:
	using container_type = ContainerType;

	/**
	 * Performs random shuffle using the specified seed
	 */
	virtual void shuffle(const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num) = 0;
	void operator() (const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num) {
		shuffle(in_container, out_container, seed, num);
	}

	/**
	 * Performs random shuffle on the entire container
	 */
	void shuffle(const ContainerType& in_container, ContainerType& out_container, uint64_t seed)
	{
		assert(in_container.size() == out_container.size());
		shuffle(in_container, out_container, seed, in_container.size());
	}

	void operator() (const ContainerType& in_container, ContainerType& out_container, uint64_t seed) {
		assert(in_container.size() == out_container.size());
		shuffle(in_container, out_container, seed, in_container.size());
	}

	/**
	 * Performs random shuffle on entire container using 0 as the seed
	 */
	void shuffle(const ContainerType& in_container, ContainerType& out_container) {
		assert(in_container.size() == out_container.size());
		shuffle(in_container, out_container, 0, in_container.size());
	}
	void operator() (const ContainerType& in_container, ContainerType& out_container) {
		assert(in_container.size() == out_container.size());
		shuffle(in_container, out_container);
	}

	virtual bool supportsInPlace() {
		return true;
	}
};