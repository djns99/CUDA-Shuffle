#include "DefaultRandomGenerator.h"
#include "nist-utils/cephes.h"
#include "nist-utils/matrix.h"
#include "shuffle/AndersonShuffle.h"
#include "shuffle/ButterflyBijectiveShuffle.h"
#include "shuffle/CzumajShuffle.h"
#include "shuffle/DartThrowing.h"
#include "shuffle/FeistelBijectiveShuffle.h"
#include "shuffle/FisherYatesShuffle.h"
#include "shuffle/LCGBijectiveShuffle.h"
#include "shuffle/MergeShuffle.h"
#include "shuffle/NoOpBijectiveShuffle.h"
#include "shuffle/RaoSandeliusShuffle.h"
#include "shuffle/SPNetworkBijectiveShuffle.h"
#include "shuffle/StdShuffle.h"
#include "shuffle/ThrustShuffle.h"
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
    ContainerType shuffled_container;
    ContainerType source_container;
    static constexpr uint64_t usable_bits = 24ull;
    // ~32 million elements in array
    static constexpr uint64_t max_num_elements = 1ull << usable_bits;
    // Fail if observed behaviour will happen < 0.1% of the time (1 in 1000)
    static constexpr double p_score_significance = 0.001;

    void SetUp()
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

    // Soboleva test functions, as described in:
    // Soboleva, M. V. (2012). The asymptotic normality of the number of congruent cycles in a random permutation,
    // Discrete Mathematics and Applications, 22(1), 91-100. doi: https://doi.org/10.1515/dma-2012-007
    template <class Container>
    std::unordered_map<uint64_t, uint64_t> cycleLengths( const Container& data )
    {
        std::unordered_map<uint64_t, uint64_t> contents_set;

        for( uint64_t i = 0; i < data.size(); i++ )
            contents_set[i] = data[i];

        std::unordered_map<uint64_t, uint64_t> result;
        while( !contents_set.empty() )
        {
            uint64_t cycle_length = 0;
            for( auto it = contents_set.begin(); it != contents_set.end(); cycle_length++ )
            {
                uint64_t next = it->second;
                contents_set.erase( it );
                it = contents_set.find( next );
            }

            result[cycle_length]++;
        }
        return result;
    }

    double sobolevaStatistic( const uint64_t n, const uint64_t d, const std::unordered_map<uint64_t, uint64_t>& cycle_lengths )
    {
        const double logn = log( n );
        const double d_div_logn = (double)d / logn;
        const double logn_div_d = logn / (double)d;

        std::vector<double> scores( d, 0 );
        for( auto cycle_pair : cycle_lengths )
            if( cycle_pair.first > d )
                scores[cycle_pair.first % d] += (double)cycle_pair.second;
        double sum_caj_sqrd = 0;
        for( auto score : scores )
        {
            const double term = score - logn_div_d;
            sum_caj_sqrd += term * term;
        }

        return d_div_logn * sum_caj_sqrd;
    }
};

template <typename ShuffleFunction>
DefaultRandomGenerator RandomnessTests<ShuffleFunction>::gen;
template <typename ShuffleFunction>
constexpr uint64_t RandomnessTests<ShuffleFunction>::usable_bits;
// ~32 million elements in array
template <typename ShuffleFunction>
constexpr uint64_t RandomnessTests<ShuffleFunction>::max_num_elements;
template <typename ShuffleFunction>
constexpr double RandomnessTests<ShuffleFunction>::p_score_significance;

using ShuffleTypes =
    ::testing::Types<StdShuffle<>, MergeShuffle<>, RaoSandeliusShuffle<>, FeistelBijectiveScanShuffle<>, ThrustShuffle<> /*, NoOpBijectiveShuffle<>*/>;
TYPED_TEST_SUITE( RandomnessTests, ShuffleTypes );