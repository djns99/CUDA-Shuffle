#include "CudaHelpers.h"
#include "GatherShuffle.h"
#include "ScatterShuffle.h"
#include "shuffle/FeistelBijectiveShuffle.h"
#include "shuffle/FisherYatesShuffle.h"
#include "shuffle/GPUSwapShuffle.h"
#include "shuffle/LCGBijectiveShuffle.h"
#include "shuffle/LubyRackoffBijectiveShuffle.h"
#include "shuffle/MergeShuffle.h"
#include "shuffle/NoOpBijectiveShuffle.h"
#include "shuffle/RaoSandeliusShuffle.h"
#include "shuffle/SPNetworkBijectiveShuffle.h"
#include "shuffle/SortShuffle.h"
#include "shuffle/StdShuffle.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <sstream>
#include <thrust/device_vector.h>
#include <vector>

using DataType = uint64_t;

template <class ShuffleFunction>
static void benchmarkScatterGather( benchmark::State& state )
{
    ShuffleFunction shuffler;
    using ContainerType = typename ShuffleFunction::container_type;

    // Shuffle second param adds 0 or 1 to compare power of two (best case) vs.
    // one above power of two (worst case)
    const uint64_t num_to_shuffle = (uint64_t)state.range( 1 ) + state.range( 0 );

    ContainerType in_container( num_to_shuffle );
    ContainerType out_container( num_to_shuffle );

    using HostVector = typename std::vector<typename ContainerType::value_type>;
    HostVector tmp( in_container.size() );
    StdShuffle<HostVector> temp_shuffler;
    thrust::sequence( tmp.begin(), tmp.end() );

    int seed = 0;
    for( auto _ : state )
    {
        state.PauseTiming();
        temp_shuffler( tmp, tmp, seed );
        thrust::copy( tmp.begin(), tmp.end(), in_container.begin() );
        checkCudaError( cudaDeviceSynchronize() );
        state.ResumeTiming();
        // Benchmarks raw gather speed of a random permutation
        shuffler( in_container, out_container, seed );
        checkCudaError( cudaDeviceSynchronize() );
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
    const uint64_t num_to_shuffle = (uint64_t)state.range( 1 ) + state.range( 0 );

    ContainerType in_container( num_to_shuffle );
    ContainerType out_container( num_to_shuffle );

    int seed = 0;
    for( auto _ : state )
    {
        shuffler( in_container, out_container, seed );
        checkCudaError( cudaDeviceSynchronize() );
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

static void argsGenerator( benchmark::internal::Benchmark* b )
{
    b->Ranges( { { 1 << 8, 1 << 29 }, { 0, 1 } } );
}

BENCHMARK_TEMPLATE( benchmarkFunction, MergeShuffle<std::vector<DataType>> )
    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, RaoSandeliusShuffle<std::vector<DataType>> )
    ->Apply( argsGenerator );
// BENCHMARK_TEMPLATE( benchmarkFunction,
// FeistelBijectiveShuffle<thrust::device_vector<DataType>> )->Apply(
// argsGenerator );
// BENCHMARK_TEMPLATE( benchmarkFunction,
// FeistelBijectiveSortShuffle<thrust::device_vector<DataType>> )
//    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, FeistelBijectiveScanShuffle<thrust::device_vector<DataType>> )
    ->Apply( argsGenerator );
// BENCHMARK_TEMPLATE( benchmarkFunction,
// SPNetworkBijectiveShuffle<thrust::device_vector<DataType>> )->Apply(
// argsGenerator );
// BENCHMARK_TEMPLATE( benchmarkFunction,
// SPNetworkBijectiveSortShuffle<thrust::device_vector<DataType>> )
//    ->Apply( argsGenerator );
// BENCHMARK_TEMPLATE( benchmarkFunction,
// SPNetworkBijectiveScanShuffle<thrust::device_vector<DataType>> )
//    ->Apply( argsGenerator );
// BENCHMARK_TEMPLATE( benchmarkFunction,
// LCGBijectiveShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator
// );
// BENCHMARK_TEMPLATE( benchmarkFunction,
// LCGBijectiveSortShuffle<thrust::device_vector<DataType>> )->Apply(
// argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, LCGBijectiveScanShuffle<thrust::device_vector<DataType>> )
    ->Apply( argsGenerator );
// BENCHMARK_TEMPLATE( benchmarkFunction,
// NoOpBijectiveShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator
// );
// BENCHMARK_TEMPLATE( benchmarkFunction,
// NoOpBijectiveSortShuffle<thrust::device_vector<DataType>> )->Apply(
// argsGenerator );
// BENCHMARK_TEMPLATE( benchmarkFunction,
// NoOpBijectiveScanShuffle<thrust::device_vector<DataType>> )->Apply(
// argsGenerator );
BENCHMARK_TEMPLATE( benchmarkScatterGather, GatherShuffle<thrust::device_vector<DataType>> )
    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkScatterGather, ScatterShuffle<thrust::device_vector<DataType>> )
    ->Apply( argsGenerator );
// BENCHMARK_TEMPLATE( benchmarkFunction,
// SortShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, StdShuffle<std::vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, FisherYatesShuffle<std::vector<DataType>> )
    ->Apply( argsGenerator );
// Too slow
// BENCHMARK_TEMPLATE( benchmarkFunction,
// LubyRackoffBijectiveShuffle<thrust::device_vector<uint64_t>> )->Apply(
// argsGenerator );
// BENCHMARK_TEMPLATE( benchmarkFunction,
// LubyRackoffBijectiveSortShuffle<thrust::device_vector<uint64_t>> )->Apply(
// argsGenerator );
// BENCHMARK_TEMPLATE( benchmarkFunction,
// LubyRackoffBijectiveScanShuffle<thrust::device_vector<uint64_t>> )->Apply(
// argsGenerator );
// BENCHMARK_TEMPLATE(benchmarkFunction,
// GPUSwapShuffle<uint64_t>)->Apply(argsGenerator);

BENCHMARK_MAIN();