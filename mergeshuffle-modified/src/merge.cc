#include "random.h"

template <class T>
void merge( T* t, unsigned int m, unsigned int n )
{
    T* u = t;
    T* v = t + m;
    T* w = t + n;

    struct random r = { 0, 0 };

    // take elements from both halves until one is exhausted

    while( 1 )
    {
        if( flip( &r ) )
        {
            if( v == w )
                break;
            swap( u, v++ );
        }
        else if( u == v )
            break;
        u++;
    }

    // finish using Fisher-Yates

    while( u < w )
    {
        unsigned int i = random_int( &r, u - t + 1 );
        swap( t + i, u++ );
    }
}
