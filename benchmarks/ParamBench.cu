#include "CudaHelpers.h"
#include "shuffle/FeistelBijectiveShuffle.h"
#include "shuffle/StdShuffle.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <sstream>
#include <vector>

using DataType = uint64_t;


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
    thrust::sequence( in_container.begin(), in_container.end(), 0 );

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
    b->Ranges( { { 1 << 24, 1 << 24 }, { 1, 1 } } );
}

template <uint64_t NumRounds, class RoundFunction>
using ParamFeistelBijectiveScanShuffle =
    BijectiveFunctionScanShuffle<FeistelBijectiveFunction<NumRounds, RoundFunction>, thrust::device_vector<uint64_t>, DefaultRandomGenerator>;
template <uint64_t NumRounds, class RoundFunction>
using HostParamFeistelBijectiveScanShuffle =
    BijectiveFunctionScanShuffle<FeistelBijectiveFunction<NumRounds, RoundFunction>, std::vector<uint64_t>, DefaultRandomGenerator>;

constexpr uint64_t target_num_rounds = 16;
BENCHMARK_TEMPLATE( benchmarkFunction,
                    ParamFeistelBijectiveScanShuffle<target_num_rounds, Taus88RanluxRoundFunction<target_num_rounds>> )
    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction,
                    ParamFeistelBijectiveScanShuffle<target_num_rounds, Taus88LCGRoundFunction<target_num_rounds>> )
    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction,
                    ParamFeistelBijectiveScanShuffle<target_num_rounds, RanluxLCGRoundFunction<target_num_rounds>> )
    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction,
                    ParamFeistelBijectiveScanShuffle<target_num_rounds, WyHashRoundFunction<target_num_rounds>> )
    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction,
                    HostParamFeistelBijectiveScanShuffle<target_num_rounds, Taus88RanluxRoundFunction<target_num_rounds>> )
    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction,
                    HostParamFeistelBijectiveScanShuffle<target_num_rounds, Taus88LCGRoundFunction<target_num_rounds>> )
    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction,
                    HostParamFeistelBijectiveScanShuffle<target_num_rounds, RanluxLCGRoundFunction<target_num_rounds>> )
    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction,
                    HostParamFeistelBijectiveScanShuffle<target_num_rounds, WyHashRoundFunction<target_num_rounds>> )
    ->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, StdShuffle<thrust::host_vector<uint64_t>> )->Apply( argsGenerator );

BENCHMARK_MAIN();