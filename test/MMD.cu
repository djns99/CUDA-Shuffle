#include "shuffle/LCGBijectiveShuffle.h"
#include "shuffle/PhiloxShuffle.h"
#include "shuffle/StdShuffle.h"
#include <algorithm>

#define LSBIT( i ) ( ( i ) & -( i ) )

class FenwickTree
{
    std::vector<size_t> data;

public:
    FenwickTree( size_t n ) : data( n )
    {
    }
    void Add( size_t i )
    {
        for( ; i < data.size(); i += LSBIT( i + 1 ) )
        {
            data[i]++;
        }
    }
    int GetCount( size_t i )
    {
        int sum = 0;
        for( ; i > 0; i -= LSBIT( i ) )
            sum += data[i - 1];
        return sum;
    }
};
template <typename T>
size_t ConcordantPairs( const std::vector<T>& x )
{
    size_t count = 0;
    FenwickTree tree( x.size() );
    for( auto x_i : x )
    {
        count += tree.GetCount( x_i );
        tree.Add( x_i );
    }
    return count;
}
template <typename T>
double MallowsKernelIdentity( const std::vector<T>& x, double lambda )
{
    auto con = ConcordantPairs( x );
    auto norm = x.size() * ( x.size() - 1 ) / 2;
    double y = 1 - ( double( con ) / norm );
    return exp( -lambda * y );
}


template <typename T>
std::vector<size_t> argsort( const std::vector<T>& array )
{
    std::vector<size_t> indices( array.size() );
    std::iota( indices.begin(), indices.end(), 0 );
    std::sort( indices.begin(), indices.end(),
               [&array]( int left, int right ) -> bool
               {
                   // sort indices according to corresponding array element
                   return array[left] < array[right];
               } );

    return indices;
}

template <typename T>
double MallowsKernel( const std::vector<T>& x, const std::vector<T>& y, double lambda )
{
    auto y_inverse = argsort( y );
    std::vector<T> z( y.size() );
    for( int i = 0; i < y.size(); i++ )
    {
        z[i] = y_inverse[x[i]];
    }
    auto concordant = ConcordantPairs( z );
    auto norm = x.size() * ( x.size() - 1 ) / 2;
    double discordant = norm - concordant;
    return exp( -lambda * ( discordant / norm ) );
}


void TestMallowsKernel()
{
    double l = 5.0;
    std::vector<int> x = { 0, 1, 2 };
    std::vector<int> y = { 2, 1, 0 };
    auto result = MallowsKernel( x, y, l );
    double expected_result = exp( -l * ( 1.0 ) );
    assert( result == expected_result );

    y = { 0, 1, 2 };
    result = MallowsKernel( x, y, l );
    expected_result = exp( -l * ( 0.0 ) );
    assert( result == expected_result );

    x = { 2, 0, 1 };
    y = { 1, 2, 0 };
    result = MallowsKernel( x, y, l );
    expected_result = exp( -l * ( 2.0 / 3 ) );
    assert( result == expected_result );
}

double MallowsExpectedValue( size_t n, double lambda )
{
    double norm = n * ( n - 1 ) / 2.0;
    double product = 1.0;
    for( auto j = 1; j <= n; j++ )
    {
        product *= ( 1.0 - exp( -lambda * j / norm ) ) / ( j * ( 1.0 - exp( -lambda / norm ) ) );
    }
    return product;
}

template <typename AlgorithmT>
double MMDSquared( size_t n, size_t num_samples, double lambda, int seed_offset = 0 )
{
    std::vector<int> in( n );
    std::iota( in.begin(), in.end(), 0 );
    double mallows_expected = 0;
    std::vector<int> x( n );
    std::vector<int> y( n );

    AlgorithmT alg;
    for( auto i = 0; i < num_samples; i++ )
    {
        alg.shuffle( in, x, i + seed_offset, n );
        i++;
        alg.shuffle( in, y, i + seed_offset, n );
        mallows_expected += MallowsKernel( x, y, lambda );
    }
    mallows_expected /= double( num_samples ) / 2;
    return abs( mallows_expected - MallowsExpectedValue( n, lambda ) );
}

template <int begin_rounds, int end_rounds>
void RunPhilox( size_t n, size_t num_samples, double lambda )
{
    if constexpr( end_rounds > begin_rounds )
    {
        RunPhilox<begin_rounds, end_rounds - 1>( n, num_samples, lambda );
    }

    double mmd =
        MMDSquared<PhiloxBijectiveScanShuffle<std::vector<int>, DefaultRandomGenerator, end_rounds>>( n, num_samples, lambda );
    printf( "VarPhilox,%f,%d,%d,%d\n", mmd, int( num_samples ), int( n ), int( end_rounds ) );
}

int main( int argc, char** argv )
{
    size_t num_samples = atoi( argv[1] );
    size_t permutation_lengths[] = { 5, 100, 1000 };
    const int begin_round = 1;
    const int end_round = 24;
    double lambda = 5;
    printf( "Algorithm,$|\\hat{\\mathrm{MMD}}^2|$,num_samples,n,Rounds\n" );
    // std::shuffle
    {
        for( auto n : permutation_lengths )
        {
            double mmd =
                MMDSquared<StdShuffle<std::vector<int>, DefaultRandomGenerator>>( n, num_samples, lambda );
            for( auto round = begin_round; round < end_round; round++ )
            {
                printf( "std::shuffle,%f,%d,%d,%d\n", mmd, int( num_samples ), int( n ), int( round ) );
            }
        }
    }

    // LCG
    {
        for( auto n : permutation_lengths )
        {
            double mmd =
                MMDSquared<LCGBijectiveScanShuffle<std::vector<int>, DefaultRandomGenerator>>( n, num_samples, lambda );
            for( auto round = begin_round; round < end_round; round++ )
            {
                printf( "LCG,%f,%d,%d,%d\n", mmd, int( num_samples ), int( n ), int( round ) );
            }
        }
    }
    // Philox
    {
        for( auto n : permutation_lengths )
        {
            RunPhilox<begin_round, end_round>( n, num_samples, lambda );
        }
    }
}