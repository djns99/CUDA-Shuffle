#include "PrefixTree.h"
#include "RandomnessTest.h"

TYPED_TEST( RandomnessTests, CountDistinctShuffles )
{
    const uint64_t shuffle_size = 100;
    const uint64_t num_samples = 1000;
    PrefixTree tree;
    for( uint64_t i = 0; i < num_samples; i++ )
    {
        this->runShuffle( shuffle_size );
        tree.add( this->shuffled_container.begin(), this->shuffled_container.begin() + shuffle_size );
    }

    uint64_t num_distinct = tree.countDistinct();
    ASSERT_EQ( num_distinct, num_samples ) << "Algorithm did not generate unique shuffles. This "
                                              "may be sporadic, but should be very unlikely";
}

TYPED_TEST( RandomnessTests, EvenDistributionAcrossLevels )
{
    const uint64_t shuffle_size = 10;
    const uint64_t num_samples = 10000;
    PrefixTree tree;
    for( uint64_t i = 0; i < num_samples; i++ )
    {
        this->runShuffle( shuffle_size );
        tree.add( this->shuffled_container.begin(), this->shuffled_container.begin() + shuffle_size );
    }

    const double expected_occurances = num_samples / shuffle_size;
    auto distribution = tree.frequencyPerLevel();
    uint64_t level_idx = 0;
    for( auto& level : distribution )
    {
        level_idx++;
        double chi_squared = 0.0;
        std::cout << "| ";
        for( uint64_t i = 0; i < shuffle_size; i++ )
        {
            auto val_it = level.find( i );
            uint64_t count = val_it == level.end() ? 0 : val_it->second;
            std::cout << i << "=" << count << " | ";
            chi_squared += pow( (double)count - expected_occurances, 2 ) / expected_occurances;
        }
        std::cout << std::endl;

        double p_score = cephes_igamc( ( shuffle_size - 1 ) / 2.0, chi_squared / 2.0 );
        EXPECT_GT( p_score, this->p_score_significance )
            << "Level " << level_idx << " failed chi squared test with chi^2=" << chi_squared
            << " giving p-value=" << p_score;
    }
}

TYPED_TEST( RandomnessTests, EvenSpacingBetweenOccurances )
{
    const uint64_t shuffle_size = 10;
    const uint64_t num_samples = 10000;

    thrust::host_vector<uint64_t> h_results( num_samples * shuffle_size, 0 );
    std::vector<std::vector<std::vector<uint64_t>>> distance_between(
        shuffle_size,
        std::vector<std::vector<uint64_t>>( shuffle_size, std::vector<uint64_t>( shuffle_size, 0 ) ) );
    for( uint64_t i = 0; i < num_samples; i++ )
    {
        this->runShuffle( shuffle_size );
        thrust::copy( this->shuffled_container.begin(), this->shuffled_container.begin() + shuffle_size,
                      h_results.begin() + i * shuffle_size );
    }

    for( uint64_t sample = 0; sample < num_samples; sample++ )
    {
        for( uint64_t i = 0; i < shuffle_size - 1; i++ )
        {
            for( uint64_t j = 1; j < shuffle_size - i; j++ )
            {
                uint64_t val_1 = h_results[sample * shuffle_size + i];
                uint64_t val_2 = h_results[sample * shuffle_size + ( i + j ) % shuffle_size];
                distance_between[val_1][val_2][j]++;
                distance_between[val_2][val_1][shuffle_size - j]++;
            }
        }
    }

    const double expected_occurances = num_samples / ( shuffle_size - 1 );
    for( uint64_t val_1 = 0; val_1 < shuffle_size; val_1++ )
    {
        for( uint64_t val_2 = val_1 + 1; val_2 < shuffle_size; val_2++ )
        {
            double chi_squared = 0.0;
            std::cout << "Distance from " << val_1 << " to " << val_2 << ": ";
            auto& distances = distance_between[val_1][val_2];
            for( uint64_t i = 1; i < shuffle_size; i++ )
            {
                std::cout << i << "=" << distances[i] << " | ";
                chi_squared += pow( (double)distances[i] - expected_occurances, 2 ) / expected_occurances;
            }
            std::cout << std::endl;

            double p_score = cephes_igamc( ( shuffle_size - 2 ) / 2.0, chi_squared / 2.0 );
            EXPECT_GT( p_score, this->p_score_significance )
                << "Values " << val_1 << " and " << val_2
                << " failed chi squared test with chi^2=" << chi_squared << " giving p-value=" << p_score;
        }
    }
}