#include "RandomnessTest.h"

TYPED_TEST( RandomnessTests, BitFrequency )
{
    // Use 256k elements to shuffle
    this->runShuffle();

    // Use the first 64k elements
    const uint64_t num_elements = 1ull << 16ull;
    const uint64_t num_possible_bits = num_elements * this->usable_bits;
    uint64_t num_ones = this->countOnes( 0, num_elements );

    int64_t final_score = ( 2 * (int64_t)num_ones ) - (int64_t)num_possible_bits;
    double p_score = erfc( (double)abs( final_score ) / (double)sqrt( num_possible_bits ) );

    std::cout << "P Score: " << p_score << std::endl;
    ASSERT_GT( p_score, this->p_score_significance ) << num_ones << "/" << num_possible_bits;
}

TYPED_TEST( RandomnessTests, BlockFrequency )
{
    // Use 256k elements to shuffle
    this->runShuffle();

    // Use the first 64k elements
    const uint64_t num_elements = 1ull << 16ull;
    const uint64_t num_possible_bits = num_elements * this->usable_bits;
    // Make sure block is a multiple of one element
    const uint64_t block_elements = 1024;
    const uint64_t block_bits = this->usable_bits * block_elements;
    const uint64_t num_blocks = num_possible_bits / block_bits;

    double sum = 0.0;
    for( uint64_t i = 0; i < num_blocks; i++ )
    {
        uint64_t num_ones = this->countOnes( i * block_elements, block_elements );
        double pi = (double)num_ones / (double)block_bits;
        double v = pi - 0.5;
        sum += v * v;
    }

    double chi_squared = 4.0 * block_bits * sum;
    double p_score = cephes_igamc( num_blocks / 2.0, chi_squared / 2.0 );

    std::cout << "P Score: " << p_score << std::endl;

    ASSERT_GT( p_score, this->p_score_significance ) << sum << "/" << block_bits;
}