#pragma once
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/subtract_with_carry_engine.h>
#include <thrust/random/xor_combine_engine.h>
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
	thrust::xor_combine_engine<thrust::linear_congruential_engine<uint64_t, 6364136223846793005U, 1442695040888963407U, 0U>, 0, thrust::ranlux48_base, 0> random_function;
};