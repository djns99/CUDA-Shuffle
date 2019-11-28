#include "RandomnessTest.h"

TYPED_TEST( RandomnessTests, BitRuns )
{
    // Use 256k elements to shuffle
    runShuffle();

    // Use the first 64k elements
    const uint64_t num_elements = 1ull << 16ull;
    const uint64_t num_total_bits = num_elements * usable_bits;

    // Pre-requisite the correct percentage of 1s
    uint64_t num_ones = countOnes( 0, num_elements );
    double percent_ones = (double)num_ones / (double)num_total_bits;
    ASSERT_LT( abs( percent_ones - 0.5 ), 2.0 / sqrt( num_total_bits ) )
        << "Failed pre-requisite test";

    uint64_t total_runs = 1;
    uint64_t last_word = 0;
    for( uint64_t i = 0; i < num_elements; i++ )
    {
        uint64_t word = shuffled_container[i];
        if( i != 0 )
            total_runs += ( word & 1ull ) != ( last_word >> 63ull );

        last_word = word;
        for( uint64_t j = 1; j < usable_bits; j++ )
        {
            if( ( ( word >> j ) & 1 ) != ( ( word >> ( j - 1 ) ) & 1 ) )
                total_runs++;
        }
    }

    double numerator = abs( total_runs - 2.0 * num_total_bits * percent_ones * ( 1 - percent_ones ) );
    double denomenator = 2.0 * percent_ones * ( 1 - percent_ones ) * sqrt( 2 * num_total_bits );
    double p_score = erfc( numerator / denomenator );

    std::cout << "P Score: " << p_score << std::endl;
    ASSERT_GT( p_score, p_score_significance );
}

TYPED_TEST( RandomnessTests, LongestRunOfOnes )
{
    runShuffle();

    // Use the first 64k elements
    const uint64_t num_elements = 1ull << 16ull;
    const uint64_t num_total_bits = num_elements * usable_bits;

    ASSERT_GT( num_total_bits, 128 );

    uint64_t bits_in_block;
    double pi[7];
    std::vector<uint64_t> bucket_counts;
    if( num_total_bits < 6272 )
    {
        bits_in_block = 8;
        for( uint64_t i = 0; i < 4; i++ )
            bucket_counts.push_back( 1 + i );
        pi[0] = 0.21484375;
        pi[1] = 0.3671875;
        pi[2] = 0.23046875;
        pi[3] = 0.1875;
    }
    else if( num_total_bits < 750000 )
    {
        bits_in_block = 128;
        for( uint64_t i = 0; i < 6; i++ )
            bucket_counts.push_back( 4 + i );
        pi[0] = 0.1174035788;
        pi[1] = 0.242955959;
        pi[2] = 0.249363483;
        pi[3] = 0.17517706;
        pi[4] = 0.102701071;
        pi[5] = 0.112398847;
    }
    else
    {
        bits_in_block = 10000;
        for( uint64_t i = 0; i < 7; i++ )
            bucket_counts.push_back( 10 + i );

        pi[0] = 0.0882;
        pi[1] = 0.2092;
        pi[2] = 0.2483;
        pi[3] = 0.1933;
        pi[4] = 0.1208;
        pi[5] = 0.0675;
        pi[6] = 0.0727;
    }

    auto bits = toVectorBool( num_total_bits );
    uint64_t num_blocks = num_total_bits / bits_in_block;
    std::vector<uint64_t> bucket_frequencies( bucket_counts.size(), 0 );
    for( uint64_t i = 0; i < num_blocks; i++ )
    {
        uint64_t max_run_length = 0;
        uint64_t current_run = 0;
        for( uint64_t j = 0; j < bits_in_block; j++ )
        {
            if( bits[i * bits_in_block + j] )
            {
                current_run++;
                if( current_run > max_run_length )
                    max_run_length = current_run;
            }
            else
            {
                current_run = 0;
            }
        }
        if( max_run_length <= bucket_counts[0] )
            bucket_frequencies[0]++;
        for( uint64_t j = 1; j < bucket_counts.size() - 1; j++ )
        {
            if( max_run_length == bucket_counts[j] )
                bucket_frequencies[j]++;
        }
        if( max_run_length >= bucket_counts[bucket_counts.size() - 1] )
            bucket_frequencies[bucket_counts.size() - 1]++;
    }

    double chi2 = 0.0;
    for( uint64_t i = 0; i < bucket_counts.size(); i++ )
        chi2 += ( ( bucket_frequencies[i] - num_blocks * pi[i] ) *
                  ( bucket_frequencies[i] - num_blocks * pi[i] ) ) /
                ( num_blocks * pi[i] );

    double p_score = cephes_igamc( (double)( bucket_counts.size() / 2.0 ), chi2 / 2.0 );

    std::cout << "P Score: " << p_score << std::endl;
    ASSERT_GT( p_score, p_score_significance );
}