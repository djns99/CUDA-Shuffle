#include "PrefixTree.h"
#include "RandomnessTest.h"
#include "shuffle/FeistelBijectiveShuffle.h"
#include <gtest/gtest.h>
#include <thread>
#include <unordered_map>

template <class ShuffleType>
class ParameterQualityTests : public RandomnessTests<ShuffleType>
{
};

constexpr bool USE_PVALUE = true;
constexpr uint64_t NUM_SAMPLES = 500;
constexpr uint64_t SAMPLE_SIZE = 1e6;

template <uint64_t NumRounds>
using ParamFeistelBijectiveScanShuffle =
    BijectiveFunctionScanShuffle<FeistelBijectiveFunction<NumRounds, WyHashRoundFunction<NumRounds>>, thrust::host_vector<uint64_t>, DefaultRandomGenerator>;

template <uint64_t NumRounds, class RoundFunction>
using ParamRoundFeistelBijectiveScanShuffle =
    BijectiveFunctionScanShuffle<FeistelBijectiveFunction<NumRounds, RoundFunction>, thrust::host_vector<uint64_t>, DefaultRandomGenerator>;

/*
 using ParameterQualityShuffleTypes = ::testing::Types<StdShuffle<thrust::host_vector<uint64_t>>,
                                                      ParamFeistelBijectiveScanShuffle<1>,
                                                      ParamFeistelBijectiveScanShuffle<2>,
                                                      ParamFeistelBijectiveScanShuffle<3>,
                                                      ParamFeistelBijectiveScanShuffle<4>,
                                                      ParamFeistelBijectiveScanShuffle<5>,
                                                      ParamFeistelBijectiveScanShuffle<6>,
                                                      ParamFeistelBijectiveScanShuffle<7>,
                                                      ParamFeistelBijectiveScanShuffle<8>,
                                                      ParamFeistelBijectiveScanShuffle<9>,
                                                      ParamFeistelBijectiveScanShuffle<10>,
                                                      ParamFeistelBijectiveScanShuffle<11>,
                                                      ParamFeistelBijectiveScanShuffle<12>,
                                                      ParamFeistelBijectiveScanShuffle<13>,
                                                      ParamFeistelBijectiveScanShuffle<14>,
                                                      ParamFeistelBijectiveScanShuffle<15>,
                                                      ParamFeistelBijectiveScanShuffle<16>,
                                                      ParamFeistelBijectiveScanShuffle<17>,
                                                      ParamFeistelBijectiveScanShuffle<18>,
                                                      ParamFeistelBijectiveScanShuffle<19>,
                                                      ParamFeistelBijectiveScanShuffle<20>,
                                                      ParamFeistelBijectiveScanShuffle<21>,
                                                      ParamFeistelBijectiveScanShuffle<22>,
                                                      ParamFeistelBijectiveScanShuffle<23>,
                                                      ParamFeistelBijectiveScanShuffle<24>,
                                                      ParamFeistelBijectiveScanShuffle<25>,
                                                      ParamFeistelBijectiveScanShuffle<26>,
                                                      ParamFeistelBijectiveScanShuffle<27>,
                                                      ParamFeistelBijectiveScanShuffle<28>,
                                                      ParamFeistelBijectiveScanShuffle<29>,
                                                      ParamFeistelBijectiveScanShuffle<30>,
                                                      ParamFeistelBijectiveScanShuffle<31>,
                                                      ParamFeistelBijectiveScanShuffle<32>

                                                      >;
*/

constexpr uint64_t target_num_rounds = 16;
using ParameterQualityShuffleTypes =
    ::testing::Types<ParamRoundFeistelBijectiveScanShuffle<target_num_rounds, Taus88RanluxRoundFunction<target_num_rounds>>,
                     ParamRoundFeistelBijectiveScanShuffle<target_num_rounds, Taus88LCGRoundFunction<target_num_rounds>>,
                     ParamRoundFeistelBijectiveScanShuffle<target_num_rounds, RanluxLCGRoundFunction<target_num_rounds>>,
                     ParamRoundFeistelBijectiveScanShuffle<target_num_rounds, WyHashRoundFunction<target_num_rounds>>,
                     ParamRoundFeistelBijectiveScanShuffle<target_num_rounds, RC5RoundFunction<target_num_rounds>>,
                     StdShuffle<thrust::host_vector<uint64_t>>>;

TYPED_TEST_SUITE( ParameterQualityTests, ParameterQualityShuffleTypes );

uint64_t factorial( uint64_t num )
{
    uint64_t res = 1;
    for( uint64_t i = 1; i <= num; i++ )
        res *= i;
    return res;
}

template <class Vector>
uint64_t permutationToIndex( const Vector& permutation, uint64_t size )
{
    uint64_t res = 0;
    uint64_t base = 1;
    // Interpret the permutation as a number with base shuffle_size
    for( uint64_t i = 0; i < size; i++ )
    {
        res += permutation[i] * base;
        base *= size;
    }
    return res;
}

const std::vector<uint64_t>& allPermutations( uint64_t size )
{
    static std::vector<uint64_t> all;
    if( all.size() != size )
    {
        const uint64_t size_fact = factorial( size );
        all.resize( size_fact );
        all.shrink_to_fit();
        std::vector<uint64_t> items( size );
        std::iota( items.begin(), items.end(), 0 );
        for( uint64_t i = 0; i < size_fact; i++ )
        {
            all[i] = permutationToIndex( items, size );
            std::next_permutation( items.begin(), items.end() );
        }
    }
    return all;
}

void reportStats( std::vector<double>& scores )
{
    std::sort( scores.begin(), scores.end() );
    double sum = std::accumulate( scores.begin(), scores.end(), 0.0 );
    double min = scores.front();
    double max = scores.back();
    double median = ( scores[scores.size() / 2] + scores[( scores.size() + 1 ) / 2] ) / 2;
    double lquart = ( scores[( scores.size() ) / 4] + scores[( scores.size() + 3 ) / 4] ) / 2;
    double uquart = ( scores[( scores.size() * 3 ) / 4] + scores[( scores.size() * 3 + 3 ) / 4] ) / 2;
    std::cout << "Min: " << min << ", LQ: " << lquart << ", Median: " << median
              << ", UQ: " << uquart << ", Max: " << max;
    std::cout << ", Mean: " << sum / (double)scores.size() << std::endl;
}

TYPED_TEST( ParameterQualityTests, FullPermutation )
{
    const uint64_t num_loops = NUM_SAMPLES;
    const uint64_t seed_start = 0xdeadbeef;
    std::vector<double> p_scores;
    for( uint64_t loop = 0; loop < num_loops; loop++ )
    {
        const uint64_t shuffle_size = 6;
        const uint64_t num_samples = SAMPLE_SIZE;

        const uint64_t num_threads = 6;
        const uint64_t samples_per_thread = ( num_samples + ( num_threads - 1 ) ) / num_threads;

        std::vector<std::unordered_map<uint64_t, uint64_t>> results_map( num_threads );
        std::vector<std::thread> threads;
        for( uint64_t tid = 0; tid < num_threads; tid++ )
        {
            threads.emplace_back( [&, tid]() {
                auto local_shuffle = this->shuffle;
                thrust::host_vector<uint64_t> input( shuffle_size );
                thrust::host_vector<uint64_t> output( shuffle_size );

                for( uint64_t i = tid * samples_per_thread;
                     i < std::min( num_samples, samples_per_thread * ( tid + 1 ) ); i++ )
                {
                    thrust::sequence( input.begin(), input.end(), 0 );
                    local_shuffle( input, output, seed_start + loop * num_samples + i, shuffle_size );
                    const uint64_t index = permutationToIndex( output, shuffle_size );
                    results_map[tid][index]++;
                }
            } );
        }

        std::unordered_map<uint64_t, uint64_t> results;
        for( auto& thread : threads )
            thread.join();

        for( auto& res : results_map )
            for( auto& pair : res )
                results[pair.first] += pair.second;

        const uint64_t size_fact = factorial( shuffle_size );
        const double expected_occurances = num_samples / (double)size_fact;

        auto& permutations = allPermutations( shuffle_size );
        double chi_squared = 0.0;
        for( uint64_t i = 0; i < size_fact; i++ )
        {
            chi_squared += pow( results[permutations[i]] - expected_occurances, 2 ) / expected_occurances;
        }

        if( USE_PVALUE )
        {
            double p_score = cephes_igamc( (double)( size_fact - 1 ) / 2.0, chi_squared / 2.0 );
            std::cout << p_score << ',' << std::flush;
            p_scores.emplace_back( p_score );
        }
        else
        {
            std::cout << chi_squared << ',' << std::flush;
        }
    }

    std::cout << std::endl;
    if( USE_PVALUE )
        reportStats( p_scores );
}

TYPED_TEST( ParameterQualityTests, PermutationLength )
{
    const uint64_t shuffle_size = SAMPLE_SIZE;
    const uint64_t num_samples = NUM_SAMPLES;
    const uint64_t max_dimension = std::min( (uint64_t)5ull, shuffle_size );

    thrust::host_vector<uint64_t> input( shuffle_size );
    thrust::sequence( input.begin(), input.end(), 0 );
    thrust::host_vector<uint64_t> output( shuffle_size );

    std::vector<std::vector<double>> p_scores( max_dimension - 2 );

    for( uint64_t i = 0; i < num_samples; i++ )
    {
        this->shuffle( input, output, i, shuffle_size );
        auto cycle_lengths = this->cycleLengths( output );
        for( uint64_t d = 2; d < max_dimension; d++ )
        {
            double chi_squared = this->sobolevaStatistic( shuffle_size, d, cycle_lengths );
            double p_value = cephes_igamc( (double)d / 2.0, chi_squared / 2.0 );
            if( USE_PVALUE )
                p_scores[d - 2].emplace_back( p_value );
            else
                p_scores[d - 2].emplace_back( chi_squared );
        }
    }


    for( auto& d_p_score : p_scores )
    {
        for( auto score : d_p_score )
            std::cout << score << ",";
        std::cout << std::endl;
        if( USE_PVALUE )
            reportStats( d_p_score );
    }
}

TYPED_TEST( ParameterQualityTests, TurningPointCount )
{
    const uint64_t shuffle_size = SAMPLE_SIZE;
    const uint64_t num_samples = NUM_SAMPLES;

    thrust::host_vector<uint64_t> input( shuffle_size );
    thrust::sequence( input.begin(), input.end(), 0 );
    thrust::host_vector<uint64_t> output( shuffle_size );

    // Distribution expected according to
    // Hombas, V. (2004). Turning Points in Random Permutations. Teaching Statistics, 26(1), 17-19.
    const double expected = 2.0 * (double)( shuffle_size - 2 ) / 3.0;
    const double stddev = sqrt( (double)( 16 * shuffle_size - 29 ) / 90.0 );

    std::vector<double> p_scores;

    for( uint64_t i = 0; i < num_samples; i++ )
    {
        this->shuffle( input, output, i, shuffle_size );
        uint64_t num_tp = 0;
        for( uint64_t j = 1; j < shuffle_size - 1; j++ )
            // Check for turning point (i.e. both neighbours bigger or smaller)
            if( ( output[j - 1] < output[j] ) == ( output[j + 1] < output[j] ) )
                num_tp++;

        const double z_score = ( (double)num_tp - expected ) / stddev;
        const double p_val = 1.0 - erf( abs( z_score ) / sqrt( 2.0 ) );
        // std::cerr << num_tp << " vs expected " << expected << " (difference of " << ((double)num_tp - expected) << "). Produced p value of: " << p_val << std::endl;
        if( USE_PVALUE )
        {
            std::cout << p_val << "," << std::flush;
            p_scores.emplace_back( p_val );
        }
        else
        {
            std::cout << z_score << "," << std::flush;
        }
    }

    std::cout << std::endl;
    if( USE_PVALUE )
        reportStats( p_scores );
}