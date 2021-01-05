#include "random.h"
#include <functional>

static std::function<uint64_t()> randFunction;

void setMergeShuffleRand64( const std::function<uint64_t()>& function )
{
    randFunction = function;
}

// get a random 64-bit register
// Uses the "rdrand" instruction giving hardware randomness
// documentation: https://software.intel.com/en-us/articles/intel-digital-random-number-generator-drng-software-implementation-guide
static inline unsigned long rand64()
{
    unsigned long r;
    __asm__ __volatile__( "0:\n\t"
                          "rdrand %0\n\t"
                          "jnc 0b"
                          : "=r"( r )::"cc" );
    return r;
}

unsigned long mergeShuffleRand64()
{
    if( randFunction )
        return (unsigned long)randFunction();
    return rand64();
}
