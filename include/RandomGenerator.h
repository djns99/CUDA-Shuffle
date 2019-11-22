#pragma once
#include <thrust/random/linear_congruential_engine.h>
#include <random>

class DefaultRandomGenerator
{
public:
	__host__
	DefaultRandomGenerator()
		: random_function(std::random_device()())
	{
	}

	__host__ __device__
	DefaultRandomGenerator(uint64_t seed)
		: random_function(seed)
	{
	}

	__host__ __device__
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
	thrust::linear_congruential_engine<uint64_t, 6364136223846793005U, 1442695040888963407U, 0U> random_function;
};

class ConstantGenerator
{
public:
	ConstantGenerator()
		: constant(0)
	{}

	ConstantGenerator(uint64_t constant)
		: constant(constant)
	{}

	__host__ __device__
		uint64_t operator() ()
	{
		return constant;
	}
private:
	uint64_t constant;
};