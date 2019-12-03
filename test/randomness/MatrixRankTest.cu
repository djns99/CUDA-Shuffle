#include "RandomnessTest.h"
#include <atomic>
#include <thread>

TYPED_TEST( RandomnessTests, MatrixRank )
{
    this->runShuffle();

    // Use the first 64k elements
    const uint64_t num_elements = 1ull << 16ull;
    const uint64_t num_total_bits = num_elements * this->usable_bits;

    std::vector<bool> input_stream = this->toVectorBool( num_total_bits );

    const int max_rank = 32;
    const uint64_t rows = max_rank;
    const uint64_t cols = max_rank;

    uint64_t num_matrices = num_total_bits / ( rows * cols );

    ASSERT_NE( num_matrices, 0 ) << "Not enought bits to make a matrix";

    int target_rank = max_rank;
    double product = 1.0;
    for( int i = 0; i < target_rank; i++ )
        product *= ( ( 1.e0 - pow( 2, i - max_rank ) ) * ( 1.e0 - pow( 2, i - max_rank ) ) ) /
                   ( 1.e0 - pow( 2, i - target_rank ) );
    // Probability of full rank
    double p_full_rank =
        pow( 2, target_rank * ( max_rank + max_rank - target_rank ) - max_rank * max_rank ) * product;

    target_rank = rows - 1;
    product = 1;
    for( int i = 0; i < target_rank; i++ )
        product *= ( ( 1.e0 - pow( 2, i - max_rank ) ) * ( 1.e0 - pow( 2, i - max_rank ) ) ) /
                   ( 1.e0 - pow( 2, i - target_rank ) );
    // Probability of full rank-1
    double p_full_rank_minus_one =
        pow( 2, target_rank * ( max_rank + max_rank - target_rank ) - max_rank * max_rank ) * product;
    // Probability of all other ranks
    double p_other_ranks = 1 - ( p_full_rank + p_full_rank_minus_one );

    uint64_t frequency_full_rank = 0;
    uint64_t frequency_full_rank_minus_one = 0;
    auto matrix = create_matrix( rows, cols );
    for( uint64_t i = 0; i < num_matrices; i++ )
    {
        populate_matrix( rows, cols, i, input_stream, matrix );
        int rank = computeRank( rows, cols, matrix );
        assert( rank <= max_rank );
        assert( rank >= 0 );
        if( rank == max_rank )
            frequency_full_rank++;
        else if( rank == max_rank - 1 )
            frequency_full_rank_minus_one++;
    }
    uint64_t frequencey_other_ranks = num_matrices - frequency_full_rank - frequency_full_rank_minus_one;

    double chi_squared =
        ( pow( (double)frequency_full_rank - num_matrices * p_full_rank, 2 ) / (double)( num_matrices * p_full_rank ) +
          pow( (double)frequency_full_rank_minus_one - num_matrices * p_full_rank_minus_one, 2 ) /
              (double)( num_matrices * p_full_rank_minus_one ) +
          pow( (double)frequencey_other_ranks - num_matrices * p_other_ranks, 2 ) /
              (double)( num_matrices * p_other_ranks ) );

    double p_score = exp( -chi_squared / 2.e0 );

    std::cout << "P Score: " << p_score << std::endl;
    ASSERT_GT( p_score, this->p_score_significance );
}
