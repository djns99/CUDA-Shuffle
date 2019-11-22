#include "gtest/gtest.h"
#include "shuffle/FisherYatesShuffle.h"
#include "shuffle/PrimeFieldShuffle.h"

template <typename ShuffleFunction>
class FunctionalTests : public ::testing::Test {
protected:
	ShuffleFunction shuffle;

};

using ShuffleTypes = ::testing::Types<FisherYatesShuffle<>, PrimeFieldShuffle<>>;
TYPED_TEST_SUITE(FunctionalTests, ShuffleTypes);

TYPED_TEST(FunctionalTests, SameLength) {
	FAIL();
}