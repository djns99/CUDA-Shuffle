#include "CudaHelpers.h"
#include "GatherShuffle.h"
#include "ThrustInclude.h"
#include "shuffle/DartThrowing.h"
#include "shuffle/LCGBijectiveShuffle.h"
#include "shuffle/MergeShuffle.h"
#include "shuffle/PhiloxShuffle.h"
#include "shuffle/RaoSandeliusShuffle.h"
#include "shuffle/SortShuffle.h"
#include "shuffle/StdShuffle.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <sstream>
#include <vector>

// #define HOST_BENCH 1
using DataType = uint64_t;

template <class ShuffleFunction>
static void benchmarkScatterGather( benchmark::State& state )
{
    ShuffleFunction shuffler;
    using ContainerType = typename ShuffleFunction::container_type;

    // Shuffle second param adds 0 or 1 to compare power of two (best case) vs.
    // one above power of two (worst case)
    const uint64_t num_to_shuffle = (uint64_t)state.range( 1 ) + ( 1ull << (uint64_t)state.range( 0 ) );

    ContainerType in_container( num_to_shuffle );
    ContainerType out_container( num_to_shuffle );

    PhiloxBijectiveScanShuffle<ContainerType> temp_shuffler;
    thrust::sequence( out_container.begin(), out_container.end() );

    int seed = 0;
    for( auto _ : state )
    {
        state.PauseTiming();
        if( ( seed % 100 ) == 0 )
            temp_shuffler( out_container, in_container, seed );
#ifndef HOST_BENCH
        checkCudaError( cudaDeviceSynchronize() );
#endif
        state.ResumeTiming();
        // Benchmarks raw gather speed of a random permutation
        shuffler( in_container, out_container, seed );
#ifndef HOST_BENCH
        checkCudaError( cudaDeviceSynchronize() );
#endif
        seed++;
    }

    state.SetItemsProcessed( state.iterations() * num_to_shuffle );
    uint64_t log = std::log2( num_to_shuffle );
    std::stringstream s;
    s << "Shuffle 2^" << log;
    if( state.range( 1 ) )
    {
        s << " + 1";
    }
    state.SetLabel( s.str() );
}

template <class ShuffleFunction>
static void benchmarkFunction( benchmark::State& state )
{
    ShuffleFunction shuffler;
    using ContainerType = typename ShuffleFunction::container_type;

    // Shuffle second param adds 0 or 1 to compare power of two (best case) vs.
    // one above power of two (worst case)
    const uint64_t num_to_shuffle = (uint64_t)state.range( 1 ) + ( 1ull << (uint64_t)state.range( 0 ) );

    ContainerType in_container( num_to_shuffle );
    ContainerType out_container( num_to_shuffle );

    int seed = 0;
    for( auto _ : state )
    {
        shuffler( in_container, out_container, seed );
#ifndef HOST_BENCH
        checkCudaError( cudaDeviceSynchronize() );
#endif
        seed++;
    }

    state.SetItemsProcessed( state.iterations() * num_to_shuffle );
    std::stringstream s;
    s << "Shuffle 2^" << state.range( 0 );
    if( state.range( 1 ) )
    {
        s << " + 1";
    }
    state.SetLabel( s.str() );
}

static void argsGenerator( benchmark::internal::Benchmark* b )
{
    // Go up by 3 so we get both odd and even numbers of bits
    std::vector<int> logs = { 8, 11, 14, 17, 20, 23, 26, 29 };
    for( int log : logs )
    {
        b->Args( { log, 0 } );
        b->Args( { log, 1 } );
    }
}

BENCHMARK_TEMPLATE( benchmarkFunction, PhiloxBijectiveScanShuffle<> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, BasicPhiloxBijectiveScanShuffle<> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, TwoPassPhiloxBijectiveScanShuffle<> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, LCGBijectiveScanShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );

BENCHMARK_TEMPLATE( benchmarkFunction, PhiloxBijectiveScanShuffle<thrust::tbb::vector<DataType>> )->Apply( argsGenerator );

BENCHMARK_TEMPLATE( benchmarkFunction, LCGBijectiveScanShuffle<thrust::tbb::vector<DataType>> )->Apply( argsGenerator );

BENCHMARK_TEMPLATE( benchmarkFunction, DartThrowing<thrust::device_vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, SortShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );

BENCHMARK_TEMPLATE( benchmarkFunction, HostDartThrowing<std::vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, MergeShuffle<std::vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, RaoSandeliusShuffle<std::vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, StdShuffle<std::vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, SortShuffle<thrust::host_vector<DataType>> )->Apply( argsGenerator );

#ifndef HOST_BENCH
BENCHMARK_TEMPLATE( benchmarkScatterGather, GatherShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );
#endif
BENCHMARK_TEMPLATE( benchmarkScatterGather, GatherShuffle<thrust::host_vector<DataType>> )->Apply( argsGenerator );

BENCHMARK_MAIN();