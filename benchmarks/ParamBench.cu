#include "CudaHelpers.h"
#include "shuffle/FeistelBijectiveShuffle.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <sstream>
#include <vector>

using DataType = uint64_t;


template <uint64_t NumRounds>
using ParamFeistelBijectiveScanShuffle =
    BijectiveFunctionScanShuffle<FeistelBijectiveFunction<NumRounds>, thrust::device_vector<uint64_t>, DefaultRandomGenerator>;


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

BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<1> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<2> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<3> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<4> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<5> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<6> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<7> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<8> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<9> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<10> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<11> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<12> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<13> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<14> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<15> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<16> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<17> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<18> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<19> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<20> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<21> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<22> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<23> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<24> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<25> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<26> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<27> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<28> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<29> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<30> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<31> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<32> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<64> )->Apply( argsGenerator );
BENCHMARK_TEMPLATE( benchmarkFunction, ParamFeistelBijectiveScanShuffle<128> )->Apply( argsGenerator );

BENCHMARK_MAIN();