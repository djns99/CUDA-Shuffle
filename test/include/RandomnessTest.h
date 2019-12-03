#include "DefaultRandomGenerator.h"
#include "EmptyShuffle.h"
#include "RandomReverseShuffle.h"
#include "nist-utils/cephes.h"
#include "nist-utils/matrix.h"
#include "shuffle/FeistelBijectiveShuffle.h"
#include "shuffle/FisherYatesShuffle.h"
#include "shuffle/PrimeFieldBijectiveShuffle.h"
#include "shuffle/PrimeFieldSortShuffle.h"
#include "shuffle/SPNetworkBijectiveShuffle.h"
#include "shuffle/SortShuffle.h"
#include "gtest/gtest.h"
#include <cmath>
#ifdef _MSC_VER
#include <intrin.h>
#define popcnt64 __popcnt64
#else
#define popcnt64 __builtin_popcountll
#endif
#include <numeric>

template <typename ShuffleFunction>
class RandomnessTests : public ::testing::Test
{
protected:
    ShuffleFunction shuffle;
    static DefaultRandomGenerator gen;
    typedef typename ShuffleFunction::Shuffle::container_type ContainerType;
    static ContainerType shuffled_container;
    static ContainerType source_container;
    static constexpr uint64_t usable_bits = 24ull;
    // ~32 million elements in array
    static constexpr uint64_t max_num_elements = 1ull << usable_bits;
    static constexpr double p_score_significance = 0.01;

    static void SetUpTestCase()
    {
        source_container = ContainerType( max_num_elements, 0 );
        shuffled_container = ContainerType( max_num_elements, 0 );
        thrust::sequence( source_container.begin(), source_container.end(), 0 );
    }

    void runShuffle()
    {
        shuffle( source_container, shuffled_container, gen() );
    }

    void runShuffle( uint64_t count )
    {
        shuffle( source_container, shuffled_container, gen(), count );
    }

    uint64_t countOnes( uint64_t start_index, uint64_t num_elements )
    {
        uint64_t num_ones = 0;
        for( uint64_t j = 0; j < num_elements; j++ )
            num_ones += popcnt64( shuffled_container[start_index + j] );
        return num_ones;
    }

    uint64_t ceil_div( uint64_t numerator, uint64_t divisor )
    {
        return ( numerator + ( divisor - 1 ) ) / divisor;
    }

    std::vector<bool> toVectorBool( uint64_t num_bits )
    {
        const uint64_t num_elements = ceil_div( num_bits, usable_bits );
        std::vector<bool> packed( num_elements * usable_bits, 0 );
        for( uint64_t i = 0; i < num_elements; i++ )
        {
            uint64_t word = shuffled_container[i];
            for( uint64_t j = 0; j < usable_bits; j++ )
            {
                packed[i * usable_bits + j] = ( word >> j ) & 1;
            }
        }
        packed.resize( num_bits );
        return packed;
    }
};

template <typename ShuffleFunction>
DefaultRandomGenerator RandomnessTests<ShuffleFunction>::gen;
template <typename ShuffleFunction>
RandomnessTests<ShuffleFunction>::ContainerType RandomnessTests<ShuffleFunction>::shuffled_container;
template <typename ShuffleFunction>
RandomnessTests<ShuffleFunction>::ContainerType RandomnessTests<ShuffleFunction>::source_container;
template <typename ShuffleFunction>
constexpr uint64_t RandomnessTests<ShuffleFunction>::usable_bits;
// ~32 million elements in array
template <typename ShuffleFunction>
constexpr uint64_t RandomnessTests<ShuffleFunction>::max_num_elements;
template <typename ShuffleFunction>
constexpr double RandomnessTests<ShuffleFunction>::p_score_significance;

using ShuffleTypes = ::testing::Types<FisherYatesShuffle<>, SPNetworkBijectiveShuffle<>>;
TYPED_TEST_SUITE( RandomnessTests, ShuffleTypes );