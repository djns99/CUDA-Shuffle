/*

Test Methodology

The tests are derived from:
https://csrc.nist.gov/projects/random-bit-generation/documentation-and-software
as described in
https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

Which is the NIST-STS (NIST SP 800-22) standard for testing randomness
These tests are intended to test random number generators.

In order to test the effectiveness of the shuffling algorithm
I have observed that a random permutation of an array of length N
that is initialised with sequentially increasing values (i.e. 0..N-1)
is equivalent to a random number generator with period N and no replacements

As a result the reference container for these tests 
*/

#include "gtest/gtest.h"
#include "shuffle/FisherYatesShuffle.h"
#include "shuffle/PrimeFieldSortShuffle.h"
#include "shuffle/PrimeFieldBijectiveShuffle.h"
#include "shuffle/SortShuffle.h"
#include "DefaultRandomGenerator.h"
#include "EmptyShuffle.h"
#include "nist-utils/cephes.h"
#include <numeric>
#include <cmath>
#include <intrin.h>



template <typename ShuffleFunction>
class RandomnessTests : public ::testing::Test {
protected:
	ShuffleFunction shuffle;
	static DefaultRandomGenerator gen;
	using ContainerType = ShuffleFunction::Shuffle::container_type;
	static ContainerType shuffled_container;
	static ContainerType source_container;
	static constexpr uint64_t usable_bits = 24ull;
	// ~32 million elements in array
	static constexpr uint64_t max_num_elements = 1ull << usable_bits;
	static constexpr double p_score_significance = 0.01;

	static void SetUpTestCase()
	{
		source_container = ContainerType(max_num_elements, 0);
		shuffled_container = ContainerType(max_num_elements, 0);
		thrust::sequence(source_container.begin(), source_container.end(), 0);
	}

	void runShuffle()
	{
		shuffle(source_container, shuffled_container, gen());
	}

	uint64_t countOnes(uint64_t start_index, uint64_t num_elements)
	{
		uint64_t num_ones = 0;
		for (uint64_t j = 0; j < num_elements; j++)
			num_ones += __popcnt64(shuffled_container[start_index + j]);
		return num_ones;
	}

	uint64_t ceil_div(uint64_t numerator, uint64_t divisor)
	{
		return (numerator + (divisor - 1)) / divisor;
	}

	std::vector<bool> toVectorBool(uint64_t num_bits)
	{
		const uint64_t num_elements = ceil_div(num_bits, usable_bits);
		std::vector<bool> packed(num_elements * usable_bits, 0);
		for (uint64_t i = 0; i < num_elements; i++)
		{
			uint64_t word = shuffled_container[i];
			for (uint64_t j = 0; j < usable_bits; j++)
			{
				packed[i * usable_bits + j] = (word >> j) & 1;
			}
		}
		packed.resize(num_bits);
		return packed;
	}

	//void
	//	LongestRunOfOnes(int n)
	//{
	//	double			pval, chi2, pi[7];
	//	int				run, v_n_obs, N, i, j, K, M, V[7];
	//	unsigned int	nu[7] = { 0, 0, 0, 0, 0, 0, 0 };

	//	if (n < 128) {
	//		fprintf(stats[TEST_LONGEST_RUN], "\t\t\t  LONGEST RUNS OF ONES TEST\n");
	//		fprintf(stats[TEST_LONGEST_RUN], "\t\t---------------------------------------------\n");
	//		fprintf(stats[TEST_LONGEST_RUN], "\t\t   n=%d is too short\n", n);
	//		return;
	//	}
	//	if (n < 6272) {
	//		K = 3;
	//		M = 8;
	//		V[0] = 1; V[1] = 2; V[2] = 3; V[3] = 4;
	//		pi[0] = 0.21484375;
	//		pi[1] = 0.3671875;
	//		pi[2] = 0.23046875;
	//		pi[3] = 0.1875;
	//	}
	//	else if (n < 750000) {
	//		K = 5;
	//		M = 128;
	//		V[0] = 4; V[1] = 5; V[2] = 6; V[3] = 7; V[4] = 8; V[5] = 9;
	//		pi[0] = 0.1174035788;
	//		pi[1] = 0.242955959;
	//		pi[2] = 0.249363483;
	//		pi[3] = 0.17517706;
	//		pi[4] = 0.102701071;
	//		pi[5] = 0.112398847;
	//	}
	//	else {
	//		K = 6;
	//		M = 10000;
	//		V[0] = 10; V[1] = 11; V[2] = 12; V[3] = 13; V[4] = 14; V[5] = 15; V[6] = 16;
	//		pi[0] = 0.0882;
	//		pi[1] = 0.2092;
	//		pi[2] = 0.2483;
	//		pi[3] = 0.1933;
	//		pi[4] = 0.1208;
	//		pi[5] = 0.0675;
	//		pi[6] = 0.0727;
	//	}

	//	N = n / M;
	//	for (i = 0; i < N; i++) {
	//		v_n_obs = 0;
	//		run = 0;
	//		for (j = 0; j < M; j++) {
	//			if (epsilon[i * M + j] == 1) {
	//				run++;
	//				if (run > v_n_obs)
	//					v_n_obs = run;
	//			}
	//			else
	//				run = 0;
	//		}
	//		if (v_n_obs < V[0])
	//			nu[0]++;
	//		for (j = 0; j <= K; j++) {
	//			if (v_n_obs == V[j])
	//				nu[j]++;
	//		}
	//		if (v_n_obs > V[K])
	//			nu[K]++;
	//	}

	//	chi2 = 0.0;
	//	for (i = 0; i <= K; i++)
	//		chi2 += ((nu[i] - N * pi[i]) * (nu[i] - N * pi[i])) / (N * pi[i]);

	//	double pval = cephes_igamc((double)(K / 2.0), chi2 / 2.0);
	//}
};

template <typename ShuffleFunction> DefaultRandomGenerator RandomnessTests<ShuffleFunction>::gen;
template <typename ShuffleFunction> ShuffleFunction::Shuffle::container_type RandomnessTests<ShuffleFunction>::shuffled_container;
template <typename ShuffleFunction> ShuffleFunction::Shuffle::container_type RandomnessTests<ShuffleFunction>::source_container;

using ShuffleTypes = ::testing::Types<FisherYatesShuffle<>,
	PrimeFieldSortShuffle<>,
	PrimeFieldBijectiveShuffle<>,
	SortShuffle<>,
	EmptyShuffle<>>;
TYPED_TEST_SUITE(RandomnessTests, ShuffleTypes);

TYPED_TEST(RandomnessTests, BitFrequency)
{
	// Use 256k elements to shuffle
	runShuffle();

	// Use the first 64k elements
	const uint64_t num_elements = 1ull << 16ull;
	const uint64_t num_possible_bits = num_elements * usable_bits;
	uint64_t num_ones = countOnes(0, num_elements);

	int64_t final_score = (2 * (int64_t)num_ones) - (int64_t)num_possible_bits;
	double p_score = erfc((double)abs(final_score) / (double)sqrt(num_possible_bits));

	std::cout << "P Score: " << p_score << std::endl;
	ASSERT_GT(p_score, p_score_significance);
}

TYPED_TEST(RandomnessTests, BlockFrequency)
{
	// Use 256k elements to shuffle
	runShuffle();

	// Use the first 64k elements
	const uint64_t num_elements = 1ull << 16ull;
	const uint64_t num_possible_bits = num_elements * usable_bits;
	// Make sure block is a multiple of one element
	const uint64_t block_elements = 1024;
	const uint64_t block_bits = usable_bits * block_elements;
	const uint64_t num_blocks = num_possible_bits / block_bits;
	
	double sum = 0.0;
	for (uint64_t i = 0; i < num_blocks; i++)
	{
		uint64_t num_ones = countOnes(i * block_elements, block_elements);
		double pi = (double)num_ones / (double)block_bits;
		double v = pi - 0.5;
		sum += v * v;
	}

	double chi_squared = 4.0 * block_bits * sum;
	double p_score = cephes_igamc(num_blocks / 2.0, chi_squared / 2.0);

	std::cout << "P Score: " << p_score << std::endl;

	ASSERT_GT(p_score, p_score_significance);
}

TYPED_TEST(RandomnessTests, BitRuns)
{
	// Use 256k elements to shuffle
	runShuffle();

	// Use the first 64k elements
	const uint64_t num_elements = 1ull << 16ull;
	const uint64_t num_total_bits = num_elements * usable_bits;

	// Pre-requisite the correct percentage of 1s
	uint64_t num_ones = countOnes(0, num_elements);
	double percent_ones = (double)num_ones / (double)num_total_bits;
	ASSERT_LT( abs(percent_ones - 0.5), 2.0 / sqrt(num_total_bits) ) << "Failed pre-requisite test";

	uint64_t total_runs = 1;
	uint64_t last_word = 0;
	for (uint64_t i = 0; i < num_elements; i++)
	{
		uint64_t word = shuffled_container[i];
		if (i != 0)
			total_runs += (word & 1ull) != (last_word >> 63ull);

		last_word = word;
		for (uint64_t j = 1; j < usable_bits; j++)
		{
			if (((word >> j) & 1) != ((word >> (j-1)) & 1))
				total_runs++;
		}
	}

	double numerator = abs(total_runs - 2.0 * num_total_bits * percent_ones * (1 - percent_ones));
	double denomenator = 2.0 * percent_ones * (1 - percent_ones) * sqrt(2 * num_total_bits);
	double p_score = erfc(numerator/denomenator);


	std::cout << "P Score: " << p_score << std::endl;
	ASSERT_GT(p_score, p_score_significance);
}