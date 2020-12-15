#include "PrefixTree.h"
#include "RandomnessTest.h"
#include "shuffle/FeistelBijectiveShuffle.h"
#include <condition_variable>
#include <gtest/gtest.h>
#include <thread>

template <class ShuffleType>
class ParameterQualityTests : public RandomnessTests<ShuffleType>
{
};

template <uint64_t NumRounds>
using ParamFeistelBijectiveScanShuffle =
    BijectiveFunctionScanShuffle<FeistelBijectiveFunction<NumRounds>, thrust::host_vector<uint64_t>, DefaultRandomGenerator>;

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
                                                      ParamFeistelBijectiveScanShuffle<32>>;
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
    const uint64_t num_loops = 50;
    std::vector<double> p_scores( num_loops, 0.0 );
    std::vector<std::thread> threads;
    uint64_t count = 6;
    std::mutex lock;
    std::condition_variable cv;
    for( uint64_t j = 0; j < num_loops; j++ )
    {
        threads.emplace_back( [&, j, this] {
            auto local_shuffle = this->shuffle;
            const uint64_t shuffle_size = 6;
            const uint64_t num_samples = 1e6;
            thrust::host_vector<uint64_t> input( shuffle_size );
            thrust::host_vector<uint64_t> output( shuffle_size );

            std::unordered_map<uint64_t, uint64_t> results;
            for( uint64_t i = 0; i < num_samples; i++ )
            {
                thrust::sequence( input.begin(), input.end(), 0 );
                local_shuffle( input, output, i + j * num_samples, shuffle_size );
                results[permutationToIndex( output, shuffle_size )]++;
            }

            const uint64_t size_fact = factorial( shuffle_size );
            const double expected_occurances = num_samples / (double)size_fact;

            auto& permutations = allPermutations( shuffle_size );
            double chi_squared = 0.0;
            for( uint64_t i = 0; i < size_fact; i++ )
            {
                chi_squared += pow( results[permutations[i]] - expected_occurances, 2 ) / expected_occurances;
            }

            double p_score = cephes_igamc( (double)( size_fact - 1 ) / 2.0, chi_squared / 2.0 );
            std::cout << p_score << ',' << std::flush;

            p_scores[j] = p_score;

            std::lock_guard<std::mutex> g( lock );
            count++;
            cv.notify_one();
        } );

        std::unique_lock<std::mutex> g( lock );
        cv.wait( g, [&]() { return count > 0; } );
        count--;
    }

    for( auto& thread : threads )
        thread.join();
    std::cout << std::endl;
    reportStats( p_scores );
}