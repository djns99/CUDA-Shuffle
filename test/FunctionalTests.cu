#include "gtest/gtest.h"
#include "shuffle/FisherYatesShuffle.h"
#include "shuffle/PrimeFieldShuffle.h"
#include "shuffle/SortShuffle.h"
#include <numeric>

template <typename ShuffleFunction>
class FunctionalTests : public ::testing::Test {
protected:
	ShuffleFunction shuffle;
	DefaultRandomGenerator gen;
	using ContainerType = ShuffleFunction::Shuffle::container_type;
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
			ASSERT_EQ(a[i], b[i]);
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
};

using ShuffleTypes = ::testing::Types<FisherYatesShuffle<>,
									  PrimeFieldShuffle<>,
	                                  SortShuffle<>, 
									  FisherYatesShuffle<std::vector<uint64_t>, ConstantGenerator>,
	                                  PrimeFieldShuffle<thrust::device_vector<uint64_t>, ConstantGenerator>,
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