#pragma once
#include "DefaultRandomGenerator.h"
#include "ThrustInclude.h"
#include "shuffle/Shuffle.h"
#include <cuda.h>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <numeric>

namespace BijectiveScanFuncs {
template <bool device> struct cached_allocator;

template <> struct cached_allocator<true> {
  typedef char value_type;

  thrust::device_vector<char> memory;

  char *allocate(std::ptrdiff_t num_bytes) {
    memory.resize(num_bytes);
    return memory.data().get();
  }

  void deallocate(char *ptr, size_t) {}
};

template <> struct cached_allocator<false> {
  typedef char value_type;

  thrust::host_vector<char> memory;

  char *allocate(std::ptrdiff_t num_bytes) {
    memory.resize(num_bytes);
    return memory.data();
  }

  void deallocate(char *ptr, size_t) {}
};

struct KeyFlagTuple {
  uint64_t key;
  uint64_t flag;
};

struct ScanOp {
  __host__ __device__ KeyFlagTuple operator()(const KeyFlagTuple &a,
                                              const KeyFlagTuple &b) const {
    return {b.key, a.flag + b.flag};
  }
};

template <typename InputIterT, typename OutputIterT>
struct WritePermutationFunctor {
  uint64_t m;
  InputIterT in;
  OutputIterT out;
  __host__ __device__ size_t operator()(KeyFlagTuple x) {
    if (x.key < m) {
      // -1 because inclusive scan
      out[x.flag - 1] = in[x.key];
    }
    return 0; // Discarded
  }
};

template <class Function> struct MakeTupleFunctor {
  uint64_t m;
  Function mapping_function;
  MakeTupleFunctor(uint64_t m, Function mapping_function)
      : m(m), mapping_function(mapping_function) {}
  __host__ __device__ KeyFlagTuple operator()(uint64_t idx) const {
    auto gather_key = mapping_function(idx);
    return KeyFlagTuple{gather_key, gather_key < m};
  }
};

} // namespace BijectiveScanFuncs

// Fastest implementation
template <class BijectiveFunction,
          class ContainerType = thrust::device_vector<uint64_t>,
          class RandomGenerator = DefaultRandomGenerator>
class BijectiveFunctionScanShuffle
    : public Shuffle<ContainerType, RandomGenerator> {
  constexpr static bool device = std::is_same<
      ContainerType,
      thrust::device_vector<typename ContainerType::value_type,
                            typename ContainerType::allocator_type>>::value;
  BijectiveScanFuncs::cached_allocator<device> alloc;
  thrust::host_vector<BijectiveScanFuncs::KeyFlagTuple> tuple;

public:
  void shuffle(const ContainerType &in_container, ContainerType &out_container,
               uint64_t seed, uint64_t num) override {
    using namespace BijectiveScanFuncs;
    assert(&in_container != &out_container);

    RandomGenerator random_function(seed);
    BijectiveFunction mapping_function;
    mapping_function.init(num, random_function);
    uint64_t capacity = mapping_function.getMappingRange();

    thrust::counting_iterator<uint64_t> indices(0);
    size_t m = num;

    WritePermutationFunctor<decltype(in_container.begin()),
                            decltype(out_container.begin())>
        write_functor{m, in_container.begin(), out_container.begin()};
    auto output_it = thrust::make_transform_output_iterator(
        thrust::discard_iterator<uint64_t>(), write_functor);
    if constexpr (device) {
      thrust::transform_iterator<MakeTupleFunctor<BijectiveFunction>,
                                 decltype(indices), KeyFlagTuple>
          tuple_it(indices,
                   MakeTupleFunctor<BijectiveFunction>(m, mapping_function));
      thrust::inclusive_scan(thrust::cuda::par(alloc), tuple_it,
                             tuple_it + capacity, output_it, ScanOp());
    } else {
      if (tuple.size() < capacity) {
        tuple.resize(capacity);
      }
      // Need to transform exactly once since computation is the bottleneck
      thrust::transform(
          thrust::tbb::par, indices, indices + capacity, tuple.begin(),
          MakeTupleFunctor<BijectiveFunction>(m, mapping_function));
      // Explicitly call TBB scan to ensure parallel operation
      thrust::system::tbb::detail::inclusive_scan(
          thrust::tbb::par, tuple.begin(), tuple.begin() + capacity, output_it,
          ScanOp());
    }
  }

  bool supportsInPlace() const override { return false; }
};

// Most basic implementation. For comparison purposes only.
template <class BijectiveFunction,
          class ContainerType = thrust::device_vector<uint64_t>,
          class RandomGenerator = DefaultRandomGenerator>
class BasicBijectiveFunctionScanShuffle
    : public Shuffle<ContainerType, RandomGenerator> {
  mgpu::standard_context_t context;

public:
  void shuffle(const ContainerType &in_container, ContainerType &out_container,
               uint64_t seed, uint64_t num) override {
    using namespace BijectiveScanFuncs;
    assert(&in_container != &out_container);

    RandomGenerator random_function(seed);
    BijectiveFunction mapping_function;
    mapping_function.init(num, random_function);
    uint64_t capacity = mapping_function.getMappingRange();
    auto m = num;
    thrust::counting_iterator<uint64_t> indices(0);
    thrust::device_vector<uint64_t> bijection(capacity);
    thrust::transform(indices, indices + capacity, bijection.begin(),
                      mapping_function);
    thrust::device_vector<uint64_t> compacted_bijection(num);
    auto d_bijection = bijection.data();
    auto compact = mgpu::transform_compact(capacity, context);
    int stream_count = compact.upsweep(
        [=] __device__(uint64_t idx) { return d_bijection[idx] < m; });
    auto d_compacted_bijection = compacted_bijection.data();
    compact.downsweep([=] __device__(int dest_index, int source_index) {
      d_compacted_bijection[dest_index] = d_bijection[source_index];
    });
    thrust::gather(compacted_bijection.begin(), compacted_bijection.end(),
                   in_container.begin(), out_container.begin());
  }

  bool supportsInPlace() const override { return false; }
};

// Scan with two passes. For comparison purposes only.
template <class BijectiveFunction,
          class ContainerType = thrust::device_vector<uint64_t>,
          class RandomGenerator = DefaultRandomGenerator>
class MGPUBijectiveFunctionScanShuffle
    : public Shuffle<ContainerType, RandomGenerator> {
  mgpu::standard_context_t context;

public:
  void shuffle(const ContainerType &in_container, ContainerType &out_container,
               uint64_t seed, uint64_t num) override {
    using namespace BijectiveScanFuncs;
    assert(&in_container != &out_container);

    RandomGenerator random_function(seed);
    BijectiveFunction mapping_function;
    mapping_function.init(num, random_function);
    uint64_t capacity = mapping_function.getMappingRange();

    size_t m = num;

    WritePermutationFunctor<decltype(in_container.begin()),
                            decltype(out_container.begin())>
        write_functor{m, in_container.begin(), out_container.begin()};
    auto output_it = thrust::make_transform_output_iterator(
        thrust::discard_iterator<uint64_t>(), write_functor);
    MakeTupleFunctor<BijectiveFunction> f(m, mapping_function);
    auto MGpuScanOp = [=] __device__(const KeyFlagTuple &a,
                                     const KeyFlagTuple &b) {
      return KeyFlagTuple{a.key, a.flag + b.flag};
    };
    mgpu::transform_scan<KeyFlagTuple, mgpu::scan_type_inc>(
        f, capacity, output_it, MGpuScanOp,
        thrust::discard_iterator<KeyFlagTuple>(), context);
  }

  bool supportsInPlace() const override { return false; }
};