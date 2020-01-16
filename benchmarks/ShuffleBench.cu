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
#include "CudaHelpers.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <sstream>
#include <thrust/device_vector.h>
#include <vector>

using DataType = uint64_t;

template <class ShuffleFunction>
static void benchmarkFunction( benchmark::State& state )
{
    ShuffleFunction shuffler;
    using ContainerType = typename ShuffleFunction::container_type;
    constexpr bool isScatterGather = std::is_same<ShuffleFunction, GatherShuffle<thrust::device_vector<DataType>>>::value || std::is_same<ShuffleFunction, ScatterShuffle<thrust::device_vector<DataType>>>::value;

    // Shuffle second param adds 0 or 1 to compare power of two (best case) vs. one above power of two (worst case)
    const uint64_t num_to_shuffle = (uint64_t)state.range( 1 ) + state.range( 0 );

    ContainerType in_container( num_to_shuffle );
    ContainerType out_container( num_to_shuffle );

    if ( isScatterGather )
    {
        thrust::host_vector<uint64_t> h_gather_container( num_to_shuffle );
        thrust::device_vector<uint64_t> d_gather_container( num_to_shuffle );
        StdShuffle<thrust::host_vector<uint64_t>> temp_shuffler;
        std::iota( h_gather_container.begin(), h_gather_container.end(), 0 );

        int seed = 0;
        for( auto _ : state )
        {
            state.PauseTiming();
            temp_shuffler( h_gather_container, h_gather_container, seed);
            thrust::copy( h_gather_container.begin(), h_gather_container.end(), in_container.begin() );
            state.ResumeTiming();
            // Benchmarks raw gather speed of a random permutation
            shuffler( in_container, out_container, seed, in_container.size() );
            checkCudaError( cudaDeviceSynchronize() );
            seed++;
        }
    }
    else
    {
        int seed = 0;
        for( auto _ : state )
        {
            shuffler( in_container, out_container, seed );
            checkCudaError( cudaDeviceSynchronize() );
            seed++;
        }
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
    b->Ranges( { { 1 << 8, 1 << 27 }, { 0, 1 } } );
}

BENCHMARK_TEMPLATE( benchmarkFunction, MergeShuffle<std::vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, RaoSandeliusShuffle<std::vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, FeistelBijectiveShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, FeistelBijectiveSortShuffle<thrust::device_vector<DataType>> )
    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, FeistelBijectiveScanShuffle<thrust::device_vector<DataType>> )
    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, SPNetworkBijectiveShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, SPNetworkBijectiveSortShuffle<thrust::device_vector<DataType>> )
    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, SPNetworkBijectiveScanShuffle<thrust::device_vector<DataType>> )
    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, LCGBijectiveShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, LCGBijectiveSortShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, LCGBijectiveScanShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, NoOpBijectiveShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, NoOpBijectiveSortShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, NoOpBijectiveScanShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, GatherShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ScatterShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, SortShuffle<thrust::device_vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, StdShuffle<std::vector<DataType>> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, FisherYatesShuffle<std::vector<DataType>> )->Apply( argsGenerator );
// Too slow
// BENCHMARK_TEMPLATE( benchmarkFunction, LubyRackoffBijectiveShuffle<thrust::device_vector<uint64_t>> )->Apply( argsGenerator );
// BENCHMARK_TEMPLATE( benchmarkFunction, LubyRackoffBijectiveSortShuffle<thrust::device_vector<uint64_t>> )->Apply( argsGenerator );
// BENCHMARK_TEMPLATE( benchmarkFunction, LubyRackoffBijectiveScanShuffle<thrust::device_vector<uint64_t>> )->Apply( argsGenerator );
// BENCHMARK_TEMPLATE(benchmarkFunction, GPUSwapShuffle<uint64_t>)->Apply(argsGenerator);


BENCHMARK_MAIN();