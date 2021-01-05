#pragma once
#include "random.h"

template <class T>
T* split( T* start, unsigned int* end )
{
    T* mid = start;
    struct random r = { 0, 0 };
    while( start < end )
    {
        if( flip( &r ) )
        {
            swap( start, mid++ );
        }
        start++;
    }
    return mid;
}
