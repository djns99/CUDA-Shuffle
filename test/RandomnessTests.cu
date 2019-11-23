/*

Test Methodology

The tests in this file come from:
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
#include <numeric>

template <typename ShuffleFunction>
class FunctionalTests : public ::testing::Test {
protected:
	static ShuffleFunction shuffle;
	static DefaultRandomGenerator gen;
	using ContainerType = ShuffleFunction::Shuffle::container_type;
	static ContainerType shuffled_container;
	static ContainerType source_container;
	// ~32 million elements in array
	static constexpr uint64_t max_num_elements = 1u << 25ul;

	static void SetUpTestCase()
	{
		source_container = ContainerType(max_num_elements, 0);
		shuffled_container = ContainerType(max_num_elements, 0);

		std::iota(source_container.begin(), source_container.end(), 0);
	}

	static void runShuffle()
	{
	
	}
};

using ShuffleTypes = ::testing::Types<FisherYatesShuffle<>,
	PrimeFieldSortShuffle<>,
	PrimeFieldBijectiveShuffle<>,
	SortShuffle<>,
	FisherYatesShuffle<std::vector<uint64_t>, ConstantGenerator>,
	PrimeFieldSortShuffle<thrust::device_vector<uint64_t>, ConstantGenerator>,
	PrimeFieldBijectiveShuffle<thrust::device_vector<uint64_t>, ConstantGenerator>,
	SortShuffle<thrust::device_vector<uint64_t>, ConstantGenerator>>;
TYPED_TEST_SUITE(FunctionalTests, ShuffleTypes);


