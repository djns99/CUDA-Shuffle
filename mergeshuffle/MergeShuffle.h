#include <cstdint>

extern void rao_sandelius_shuffle( uint64_t* t, uint64_t n );
extern void rao_sandelius_shuffle( uint32_t* t, uint64_t n );
extern void parallel_merge_shuffle( uint64_t* t, uint64_t n );
extern void parallel_merge_shuffle( uint32_t* t, uint64_t n );
