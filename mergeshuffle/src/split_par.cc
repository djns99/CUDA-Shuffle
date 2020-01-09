#include "fisher_yates.h"

#ifdef __AVX__
#include "split_vec.h"
#else
// #include "split.h"
#endif

#include <thread>
#include <stdio.h>
#include <cstdint>

extern unsigned long cutoff;
unsigned long cutoff2 = 0x100000;

template<class T>
void rao_sandelius_shuffle_thread( T* start, T* end )
{
    if( end - start <= cutoff )
    {
        fisher_yates( start, end - start ); // Small input, use Fisher-Yates
    }
    else
    {
        T* mid = split( start, end );
        if( end - start <= cutoff2 )
        {
            rao_sandelius_shuffle_thread( start, mid ); // Intermediate input,
            rao_sandelius_shuffle_thread( mid, end ); // use sequential Rao-Sandelius
        }
        else
        {
            std::thread thread( [mid, end]() {
                rao_sandelius_shuffle_thread( mid, end );
            });
            rao_sandelius_shuffle_thread( start, mid );
            thread.join();
        }
    }
}

void rao_sandelius_shuffle( uint32_t* t, uint64_t n )
{
    rao_sandelius_shuffle_thread( t, t + n );
}

void rao_sandelius_shuffle( uint64_t* t, uint64_t n )
{
    rao_sandelius_shuffle_thread( t, t + n );
}
