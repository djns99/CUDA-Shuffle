#pragma once

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/random.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/shuffle.h>

#include <thrust/system/cpp/vector.h>
#include <thrust/system/cpp/execution_policy.h>
#include <thrust/system/omp/vector.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/tbb/vector.h>
#include <thrust/system/tbb/execution_policy.h>