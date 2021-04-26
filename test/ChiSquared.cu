#include "shuffle/LCGBijectiveShuffle.h"
#include "shuffle/PhiloxShuffle.h"
#include "shuffle/StdShuffle.h"
#include <map>
#include <numeric>

size_t Factorial( int n )
{
    size_t fact = 1;
    for( int i = 1; i <= n; i++ )
    {
        fact = fact * i;
    }
    return fact;
}
double GetStatistic( std::map<std::vector<int>, int> count, size_t n, size_t num_samples )
{
    double chi_squared = 0.0;
    double expected = double( num_samples ) / Factorial( n );
    for( const auto& kv : count )
    {
        double diff = kv.second - expected;
        chi_squared += ( diff * diff ) / expected;
    }

    return chi_squared;
}

template <int begin_rounds, int end_rounds>
void RunPhilox( size_t n, size_t num_samples )
{
    if constexpr( end_rounds > begin_rounds )
    {
        RunPhilox<begin_rounds, end_rounds - 1>( n, num_samples );
    }
    std::vector<int> in( n );
    std::vector<int> out( n );
    std::iota( in.begin(), in.end(), 0 );

    std::map<std::vector<int>, int> count;
    printf( "Philox," );
    PhiloxBijectiveScanShuffle<std::vector<int>, DefaultRandomGenerator, end_rounds> alg;
    for( auto i = 0ull; i < num_samples; i++ )
    {
        alg.shuffle( in, out, i, n );
        count[out]++;
    }
    double statistic = GetStatistic( count, n, num_samples );
    printf( "%f,%d\n", statistic, end_rounds );
}

int main( int argc, char** argv )
{
    printf( "Algorithm,$\\chi^2$,Rounds\n" );
    size_t n = 5;
    const int begin_round = 14;
    const int end_round = 30;
    size_t num_samples = atoi( argv[1] );
    {
        RunPhilox<begin_round, end_round>( n, num_samples );
    }
    std::vector<int> in( n );
    std::vector<int> out( n );
    std::iota( in.begin(), in.end(), 0 );

    {
        std::map<std::vector<int>, int> count;
        StdShuffle<std::vector<int>, DefaultRandomGenerator> alg;
        for( auto i = 0ull; i < num_samples; i++ )
        {
            alg.shuffle( in, out, i, n );
            count[out]++;
        }
        double statistic = GetStatistic( count, n, num_samples );
        for( auto i = begin_round; i <= end_round; i++ )
        {
            printf( "std::shuffle,%f,%d\n", statistic, int( i ) );
        }
    }
}
