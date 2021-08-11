#!/usr/bin/env bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
cd benchmarks
export OMP_NUM_THREADS=40
./ShuffleBench --benchmark_out=../../result.json --benchmark_out_format=json --benchmark_repetitions=5
cd ../..
python bench_json_to_graph.py result.json
python plot_hypothesis_tests.py 