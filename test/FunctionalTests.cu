#include "gtest/gtest.h"
#include "shuffle/FisherYatesShuffle.h"
#include "shuffle/PrimeFieldSortShuffle.h"
#include "shuffle/PrimeFieldBijectiveShuffle.h"
#include "shuffle/SortShuffle.h"
#include <numeric>

template <typename ShuffleFunction>
class FunctionalTests : public ::testing::Test {
protected:
	ShuffleFunction shuffle;
	DefaultRandomGenerator gen;
	using ContainerType = ShuffleFunction::Shuffle::container_type;
	using RandomGenerator = ShuffleFunction::Shuffle::random_generator;
	ContainerType shuffled_container;
	ContainerType reference_container;
	uint64_t num_elements = 100;

	void SetUp()
	{
		reference_container = ContainerType(num_elements, 0);
		shuffled_container = ContainerType(num_elements, 0);

		std::iota(reference_container.begin(), reference_container.end(), 0);
	}

	static void checkSameOrder(const ContainerType& a, const ContainerType& b)
	{
		for (uint64_t i = 0; i < a.size(); i++)
		{
			ASSERT_EQ(a[i], b[i]) << "At index " << i;
		}
	}

	// Checking for same values as reference container is the main check for functional correctness
	static void checkSameValues(ContainerType& a, ContainerType& b)
	{
		// Sort both containers and check no items were lost during shuffle
		thrust::sort(a.begin(), a.end());
		thrust::sort(b.begin(), b.end());

		checkSameOrder(a, b);
	}

	bool supportsInPlace()
	{
		return shuffle.supportsInPlace();
	}

	bool isConstantRandomGenerator() {
		return std::is_same<RandomGenerator, ConstantGenerator>::value;
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

TYPED_TEST(FunctionalTests, SameLength)
{
	shuffle(reference_container, shuffled_container, gen());
	ASSERT_EQ(shuffled_container.size(), reference_container.size());
}

TYPED_TEST(FunctionalTests, SameValues)
{
	shuffle(reference_container, shuffled_container, gen());

	checkSameValues(reference_container, shuffled_container);
}

TYPED_TEST(FunctionalTests, Detereministic)
{
	ContainerType copy_container(num_elements, 0);
	shuffle(reference_container, shuffled_container, 0);
	shuffle(reference_container, copy_container, 0);

	checkSameOrder(shuffled_container, copy_container);
}

TYPED_TEST(FunctionalTests, DefaultSeedIsZero)
{
	ContainerType copy_container(num_elements, 0);
	shuffle(reference_container, shuffled_container);
	shuffle(reference_container, copy_container, 0);

	checkSameOrder(shuffled_container, copy_container);
}

TYPED_TEST(FunctionalTests, ShuffleOne)
{
	// Copy reference vector
	std::copy(reference_container.begin() + 1, reference_container.end(), shuffled_container.begin() + 1);
	// Make first location invalid
	shuffled_container[0] = num_elements + 1;
	shuffle(reference_container, shuffled_container, gen(), 1);
	checkSameOrder(reference_container, shuffled_container);
}

TYPED_TEST(FunctionalTests, ShuffleHalf)
{
	const size_t shuffle_size = num_elements / 2;
	// Copy reference vector into second half
	std::copy(reference_container.begin() + shuffle_size, reference_container.end(), shuffled_container.begin() + shuffle_size);
	shuffle(reference_container, shuffled_container, gen(), shuffle_size);

	// Check second half was not changed
	for (uint64_t i = shuffle_size; i < num_elements; i++)
	{
		ASSERT_EQ(reference_container[i], shuffled_container[i]) << "At index " << i;
	}

	checkSameValues(reference_container, shuffled_container);
}

TYPED_TEST(FunctionalTests, ShuffleInplace)
{
	if(!supportsInPlace())
	{
		GTEST_SKIP();
		return;
	}
	ContainerType copy_container(reference_container.begin(), reference_container.end());
	shuffle(reference_container, shuffled_container, 0);
	shuffle(copy_container, copy_container, 0);
	checkSameOrder(shuffled_container, copy_container);
}

TYPED_TEST(FunctionalTests, ChangesOrder)
{
	if (isConstantRandomGenerator())
	{
		GTEST_SKIP();
		return;
	}
	shuffle(reference_container, shuffled_container, gen());
	ASSERT_FALSE(std::equal(reference_container.begin(), reference_container.end(), shuffled_container.begin()));
}

TYPED_TEST(FunctionalTests, SeedsChangeOrder)
{
	if (isConstantRandomGenerator())
	{
		GTEST_SKIP();
		return;
	}
	ContainerType copy_container(num_elements, 0);
	shuffle(reference_container, shuffled_container, 0);
	shuffle(reference_container, copy_container, 1);
	// Different seeds yield different values
	ASSERT_FALSE(std::equal(reference_container.begin(), reference_container.end(), shuffled_container.begin()));
}