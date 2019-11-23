#include "gtest/gtest.h"
#include "shuffle/FisherYatesShuffle.h"
#include "shuffle/SortShuffle.h"
#include <numeric>

template <typename ShuffleFunction>
class FunctionalTests : public ::testing::Test {
protected:
	ShuffleFunction shuffle;
	DefaultRandomGenerator gen;
	using ContainerType = ShuffleFunction::Shuffle::container_type;
	ContainerType shuffled_container;
	ContainerType source_container;
	uint64_t num_elements = 100;

	void SetUp()
	{
		source_container = ContainerType(num_elements, 0);
		shuffled_container = ContainerType(num_elements, 0);

		std::iota(source_container.begin(), source_container.end(), 0);
	}

};
