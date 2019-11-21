#pragma once
#include <thrust/random/linear_congruential_engine.h>
#include <random>

class RandomGenerator {
public:
	RandomGenerator() = default;
	RandomGenerator(const RandomGenerator&) = default;
	virtual ~RandomGenerator() = default;
	__host__ __device__
	virtual uint64_t operator() () = 0;
};

class DefaultRandomGenerator : public RandomGenerator
{
public:
	DefaultRandomGenerator()
		: random_function(std::random_device()())
	{
	}

	DefaultRandomGenerator(uint64_t seed)
		: random_function(seed)
	{
	}

	DefaultRandomGenerator(DefaultRandomGenerator& other)
		: random_function(other())
	{
	}

	__host__ __device__
		uint64_t operator() ()
	{
		return random_function();
	}
private:
	thrust::minstd_rand random_function;
};