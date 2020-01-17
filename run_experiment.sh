#!/usr/bin/env bash
cd build/benchmarks
export OMP_NUM_THREADS=40
./ShuffleBench --benchmark_out=result.json --benchmark_out_format=json --benchmark_repetitions=5 #--benchmark_filter=.*Feis