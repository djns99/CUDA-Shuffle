n = 10000000

is_prime = [False, False] + [True] * (n - 1)
primes = [2]

for j in range(4, n + 1, 2):
    is_prime[j] = False

for i in range(3, n + 1, 2):
    if is_prime[i]:
        primes.append(i)
        for j in range(i * i, n + 1, i):
            is_prime[j] = False

print("#pragma once")
print("constexpr static uint64_t MAX_CACHED_PRIME = {};".format(n))
print("constexpr static uint32_t PRIME_CACHE[] = {")
print(",".join([str(x) for x in primes]))
print("};")
