#include "ConstantGenerator.h"
#include "DefaultRandomGenerator.h"
#include "shuffle/FeistelBijectiveShuffle.h"
#include "shuffle/FisherYatesShuffle.h"
#include "shuffle/StdShuffle.h"
#include "shuffle/GPUSwapShuffle.h"
#include "shuffle/PrimeFieldBijectiveShuffle.h"
#include "shuffle/PrimeFieldSortShuffle.h"
#include "shuffle/SPNetworkBijectiveShuffle.h"
#include "shuffle/SortShuffle.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <numeric>

template <typename ShuffleFunction>
class FunctionalTests : public ::testing::Test
{
public:
    ShuffleFunction shuffle;
    DefaultRandomGenerator gen;
    using ContainerType = typename ShuffleFunction::Shuffle::container_type;
    using RandomGenerator = typename ShuffleFunction::Shuffle::random_generator;
    ContainerType shuffled_container;
    ContainerType reference_container;
    uint64_t num_elements = 257;

    void SetUp()
    {
        reference_container = ContainerType( num_elements, 0 );
        shuffled_container = ContainerType( num_elements, 0 );

        std::iota( reference_container.begin(), reference_container.end(), 0 );
    }

    static void checkSameOrder( const ContainerType& a, const ContainerType& b )
    {
        for( uint64_t i = 0; i < a.size(); i++ )
        {
            ASSERT_EQ( a[i], b[i] ) << "At index " << i;
        }
    }

    // Checking for same values as reference container is the main check for functional correctness
    static void checkSameValues( ContainerType& a, ContainerType& b )
    {
        // Sort both containers and check no items were lost during shuffle
        thrust::sort( a.begin(), a.end() );
        thrust::sort( b.begin(), b.end() );

        checkSameOrder( a, b );
    }

    bool supportsInPlace()
    {
        return shuffle.supportsInPlace();
    }

    bool isConstantRandomGenerator()
    {
        return std::is_same<RandomGenerator, ConstantGenerator>::value;
    }
};

using ShuffleTypes =
    ::testing::Types<FisherYatesShuffle<>,
                     GPUSwapShuffle<>,
                     StdShuffle<>,
                     PrimeFieldBijectiveShuffle<>,
                     FeistelBijectiveShuffle<>,
                     SPNetworkBijectiveShuffle<>,
                     FisherYatesShuffle<std::vector<uint64_t>, ConstantGenerator>,
                     StdShuffle<std::vector<uint64_t>, ConstantGenerator>,
                     PrimeFieldBijectiveShuffle<thrust::device_vector<uint64_t>, ConstantGenerator>,
                     FeistelBijectiveShuffle<thrust::device_vector<uint64_t>, ConstantGenerator>,
                     SPNetworkBijectiveShuffle<thrust::device_vector<uint64_t>, ConstantGenerator>>;
TYPED_TEST_SUITE( FunctionalTests, ShuffleTypes );

TYPED_TEST( FunctionalTests, SameLength )
{
    this->shuffle( this->reference_container, this->shuffled_container, this->gen() );
    ASSERT_EQ( this->shuffled_container.size(), this->reference_container.size() );
}

TYPED_TEST( FunctionalTests, SameValues )
{
    this->shuffle( this->reference_container, this->shuffled_container, this->gen() );
    this->checkSameValues( this->reference_container, this->shuffled_container );
}

TYPED_TEST( FunctionalTests, Detereministic )
{
    typename FunctionalTests<TypeParam>::ContainerType copy_container( this->num_elements, 0 );
    this->shuffle( this->reference_container, this->shuffled_container, 0 );
    this->shuffle( this->reference_container, copy_container, 0 );

    this->checkSameOrder( this->shuffled_container, copy_container );
}

TYPED_TEST( FunctionalTests, DefaultSeedIsZero )
{
    typename FunctionalTests<TypeParam>::ContainerType copy_container( this->num_elements, 0 );
    this->shuffle( this->reference_container, this->shuffled_container );
    this->shuffle( this->reference_container, copy_container, 0 );

    this->checkSameOrder( this->shuffled_container, copy_container );
}

TYPED_TEST( FunctionalTests, ShuffleOne )
{
    // Copy reference vector
    std::copy( this->reference_container.begin() + 1, this->reference_container.end(),
               this->shuffled_container.begin() + 1 );
    // Make first location invalid
    this->shuffled_container[0] = this->num_elements + 1;
    this->shuffle( this->reference_container, this->shuffled_container, this->gen(), 1 );
    this->checkSameOrder( this->reference_container, this->shuffled_container );
}

TYPED_TEST( FunctionalTests, ShuffleHalf )
{
    const size_t shuffle_size = this->num_elements / 2;
    // Copy reference vector into second half
    std::copy( this->reference_container.begin() + shuffle_size, this->reference_container.end(),
               this->shuffled_container.begin() + shuffle_size );
    this->shuffle( this->reference_container, this->shuffled_container, this->gen(), shuffle_size );

    // Check second half was not changed
    for( uint64_t i = shuffle_size; i < this->num_elements; i++ )
    {
        ASSERT_EQ( this->reference_container[i], this->shuffled_container[i] ) << "At index " << i;
    }

    this->checkSameValues( this->reference_container, this->shuffled_container );
}

TYPED_TEST( FunctionalTests, ShuffleInplace )
{
    if( !this->supportsInPlace() )
    {
        // GTEST_SKIP();
        return;
    }
    typename FunctionalTests<TypeParam>::ContainerType copy_container( this->reference_container.begin(),
                                                                       this->reference_container.end() );
    this->shuffle( this->reference_container, this->shuffled_container, 0 );
    this->shuffle( copy_container, copy_container, 0 );
    this->checkSameOrder( this->shuffled_container, copy_container );
}

TYPED_TEST( FunctionalTests, ChangesOrder )
{
    if( this->isConstantRandomGenerator() )
    {
        // GTEST_SKIP();
        return;
    }
    this->shuffle( this->reference_container, this->shuffled_container, this->gen() );
    ASSERT_FALSE( std::equal( this->reference_container.begin(), this->reference_container.end(),
                              this->shuffled_container.begin() ) );
}

TYPED_TEST( FunctionalTests, SeedsChangeOrder )
{
    if( this->isConstantRandomGenerator() )
    {
        // GTEST_SKIP();
        return;
    }
    typename FunctionalTests<TypeParam>::ContainerType copy_container( this->num_elements, 0 );
    this->shuffle( this->reference_container, this->shuffled_container, 0 );
    this->shuffle( this->reference_container, copy_container, 1 );
    // Different seeds yield different values
    ASSERT_FALSE( std::equal( copy_container.begin(), copy_container.end(), this->shuffled_container.begin() ) );
}