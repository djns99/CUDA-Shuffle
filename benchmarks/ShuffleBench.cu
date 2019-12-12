#include <benchmark/benchmark.h>
#include <vector>
#include <thrust/device_vector.h>
#include "shuffle/FeistelBijectiveShuffle.h"
#include "shuffle/FisherYatesShuffle.h"
#include "shuffle/GPUSwapShuffle.h"
#include "shuffle/PrimeFieldBijectiveShuffle.h"
#include "shuffle/PrimeFieldSortShuffle.h"
#include "shuffle/SortShuffle.h"
#include "shuffle/SPNetworkBijectiveShuffle.h"
#include "shuffle/StdShuffle.h"


template<class ShuffleFunction>
static void benchmarkFunction(benchmark::State& state) {
    ShuffleFunction shuffler;
    using ContainerType = typename ShuffleFunction::container_type;

    // Shuffle second param adds 0 or 1 to compare power of two (best case) vs. one above power of two (worst case)
    uint64_t num_to_shuffle = (uint64_t)state.range(0) + state.range(1);

    ContainerType in_container( num_to_shuffle );
    ContainerType out_container( num_to_shuffle );

    for( auto _ : state )
    {
        shuffler( in_container, out_container );
    }

    state.SetItemsProcessed( state.iterations() * num_to_shuffle );
}

static void argsGenerator(benchmark::internal::Benchmark* b) {
    b->Ranges({{1<<8, 1<<30}, {0, 1}});
}

BENCHMARK_TEMPLATE(benchmarkFunction, StdShuffle<std::vector<uint8_t>>)->Apply(argsGenerator);
BENCHMARK_TEMPLATE(benchmarkFunction, FisherYatesShuffle<std::vector<uint8_t>>)->Apply(argsGenerator);
BENCHMARK_TEMPLATE(benchmarkFunction, FeistelBijectiveShuffle<thrust::device_vector<uint8_t>>)->Apply(argsGenerator);
BENCHMARK_TEMPLATE(benchmarkFunction, SPNetworkBijectiveShuffle<thrust::device_vector<uint8_t>>)->Apply(argsGenerator);
BENCHMARK_TEMPLATE(benchmarkFunction, PrimeFieldBijectiveShuffle<thrust::device_vector<uint8_t>>)->Apply(argsGenerator);
// Too slow
// BENCHMARK_TEMPLATE(benchmarkFunction, GPUSwapShuffle<uint8_t>)->Apply(argsGenerator);
// Too much memory
//BENCHMARK_TEMPLATE(benchmarkFunction, PrimeFieldSortShuffle<thrust::device_vector<uint8_t>>)->Apply(argsGenerator);
// BENCHMARK_TEMPLATE(benchmarkFunction, SortShuffle<thrust::device_vector<uint8_t>>)->Apply(argsGenerator);


BENCHMARK_MAIN();