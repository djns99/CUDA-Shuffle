#pragma once
#include "ThrustInclude.h"
#include "WyHash.h"
#include "shuffle/Shuffle.h"
#include <cub/device/device_radix_sort.cuh>

template <class ContainerType = thrust::device_vector<uint64_t>,
          class RandomGenerator = DefaultRandomGenerator>
class SortShuffle : public Shuffle<ContainerType, RandomGenerator> {
  constexpr static bool device = std::is_same<
      ContainerType,
      thrust::device_vector<typename ContainerType::value_type,
                            typename ContainerType::allocator_type>>::value;
  ContainerType tmp;

public:
  SortShuffle() {}

  void shuffle(const ContainerType &in_container, ContainerType &out_container,
               uint64_t seed, uint64_t num) override {
    tmp.resize(in_container.size());
    out_container = in_container;
    auto counting = thrust::make_counting_iterator(0ll);

    if constexpr (device) {
      thrust::transform(counting, counting + num, tmp.begin(),
                        [=] __device__(int64_t idx) {
                          GPURandomGenerator rng(seed, idx);
                          return rng();
                        });
      thrust::sort_by_key(tmp.begin(), tmp.begin() + num,
                          out_container.begin());
    } else {
      thrust::transform(thrust::tbb::par, counting, counting + num, tmp.begin(),
                        [=] __host__(int64_t idx) {
                          return WyHash::wyhash64_v3_pair(seed, idx);
                        });
      thrust::sort_by_key(thrust::tbb::par, tmp.begin(), tmp.begin() + num,
                          out_container.begin());
    }
  }

  bool supportsInPlace() const override { return false; }
};
