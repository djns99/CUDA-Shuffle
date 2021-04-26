#pragma once
#include "shuffle/BijectiveFunctionScanShuffle.h"

__host__ __device__ uint32_t mulhilo(uint64_t a, uint32_t b, uint32_t &hip) {
  uint64_t product = a * uint64_t(b);
  hip = product >> 32;
  return uint32_t(product);
}

template <uint64_t num_rounds = 24> class PhiloxBijectiveFunction {
public:
  template <class RandomGenerator>
  void init(uint64_t capacity, RandomGenerator &random_function) {
    uint64_t total_bits = getCipherBits(capacity);
    // Half bits rounded down
    left_side_bits = total_bits / 2;
    left_side_mask = (1ull << left_side_bits) - 1;
    // Half the bits rounded up
    right_side_bits = total_bits - left_side_bits;
    right_side_mask = (1ull << right_side_bits) - 1;
    for (int i = 0; i < num_rounds; i++) {
      key[i] = random_function();
    }
  }

  uint64_t getMappingRange() const {
    return 1ull << (left_side_bits + right_side_bits);
  }

  __host__ __device__ uint64_t operator()(const uint64_t val) const {
    uint32_t state[2] = {uint32_t(val >> right_side_bits),
                         uint32_t(val & right_side_mask)};
    for (int i = 0; i < num_rounds; i++) {
      uint32_t hi;
      uint32_t lo = mulhilo(M0, state[0], hi);
      lo = (lo << (right_side_bits - left_side_bits)) |
           state[1] >> left_side_bits;
      state[0] = ((hi ^ key[i]) ^ state[1]) & left_side_mask;
      state[1] = lo & right_side_mask;
    }
    // Combine the left and right sides together to get result
    return (uint64_t)state[0] << right_side_bits | (uint64_t)state[1];
  }

  constexpr static bool isDeterministic() { return true; }

private:
  uint64_t getCipherBits(uint64_t capacity) {
    if (capacity == 0)
      return 0;
    uint64_t i = 0;
    capacity--;
    while (capacity != 0) {
      i++;
      capacity >>= 1;
    }

    return std::max(i, uint64_t(4));
  }

  uint64_t right_side_bits;
  uint64_t left_side_bits;
  uint64_t right_side_mask;
  uint64_t left_side_mask;
  uint32_t key[num_rounds];
  static const uint64_t M0 = UINT64_C(0xD2B74407B1CE6E93);
};

template <class ContainerType = thrust::device_vector<uint64_t>,
          class RandomGenerator = DefaultRandomGenerator, uint64_t num_rounds=24>
using PhiloxBijectiveScanShuffle =
    BijectiveFunctionScanShuffle<PhiloxBijectiveFunction<num_rounds>, ContainerType,
                                 RandomGenerator>;
template <class ContainerType = thrust::device_vector<uint64_t>,
          class RandomGenerator = DefaultRandomGenerator>
using BasicPhiloxBijectiveScanShuffle =
    BasicBijectiveFunctionScanShuffle<PhiloxBijectiveFunction<>, ContainerType,
                                 RandomGenerator>;
template <class ContainerType = thrust::device_vector<uint64_t>,
          class RandomGenerator = DefaultRandomGenerator>
using TwoPassPhiloxBijectiveScanShuffle =
    MGPUBijectiveFunctionScanShuffle<PhiloxBijectiveFunction<>, ContainerType,
                                 RandomGenerator>;
