#include "random.h"
#include <immintrin.h>

#define load32( x ) _mm_loadu_ps( (float*)( x ) )
#define perm32( x, y ) _mm_permutevar_ps( ( __m128 )( x ), ( __m128i )( y ) )
#define maskstore32( x, y, z ) _mm_maskstore_ps( (float*)( x ), ( __m128i )( y ), ( __m128 )( z ) )
#define sll32( x, y ) _mm_sll_epi32( ( __m128i )( x ), ( __m128i )( y ) )
#define srl32( x, y ) _mm_srl_epi32( ( __m128i )( x ), ( __m128i )( y ) )

#define load64( x ) _mm_loadu_pd( (double*)( x ) )
#define perm64( x, y ) _mm_permutevar_pd( ( __m128d )( x ), ( __m128i )( y ) )
#define maskstore64( x, y, z ) \
    _mm_maskstore_pd( (double*)( x ), ( __m128i )( y ), ( __m128d )( z ) )
#define sll64( x, y ) _mm_sll_epi64( ( __m128i )( x ), ( __m128i )( y ) )
#define srl64( x, y ) _mm_srl_epi64( ( __m128i )( x ), ( __m128i )( y ) )

#define popcnt _mm_popcnt_u64

template <class T>
void merge( T* t, unsigned int m, unsigned int n )
{
    T* u = t;
    T* v = t + m;
    T* w = t + n;
    struct random r = { 0, 0 };

    while( true )
    {
        if constexpr( sizeof( T ) == 4 )
        {
            // draw four random bits
            consume_bits( &r, 4 );
            int p = r.x & 15;

            // 4 elements are drawn, popcnt(p) of which come from the second half
            T* uu = u + 4;
            T* vv = v + popcnt( p );

            // if proceeding would bring us too far, undo and stop
            if( uu > v || vv > w )
            {
                r.c += 4;
                break;
            }

            constexpr __v4su perm1 = { 303239696u, 1803315264u, 3166732288u, 3221225472u };
            constexpr __v4su perm2 = { 0u, 1077952576u, 2483065856u, 3918790656u };
            constexpr __v4su mask1 = { 1073741823, 54476799, 197439, 3 };
            constexpr __v4su mask2 = { 858993459, 252645135, 16711935, 65535 };

            __v4si pp = { 2 * p, 2 * p, 2 * p, 2 * p };
            __m128 a = perm32( load32( u ), srl32( perm1, pp ) );
            __m128 b = perm32( load32( v ), srl32( perm2, pp ) );
            maskstore32( u, sll32( mask2, pp ), b );
            maskstore32( v, sll32( mask1, pp ), a );

            u = uu;
            v = vv;
            r.x >>= 4;
        }
        else
        {
            // draw two random bits
            consume_bits( &r, 2 );
            int p = r.x & 0x3;

            // 2 elements are drawn, popcnt(p) of which come from the second half
            T* uu = u + 2;
            T* vv = v + popcnt( p );

            // if proceeding would bring us too far, undo and stop
            if( uu > v || vv > w )
            {
                r.c += 2;
                break;
            }

            constexpr __v2di perm1 = { 0x4, 0x8 };
            constexpr __v2di perm2 = { 0x0, 0x8 };

            constexpr __v2di maskA = { 0x7FFFFFFFFFFFFFFF, 0x1FFFFFFFFFFFFFFF };
            constexpr __v2di maskB = { 0x5FFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF };

            __v4si pp = { p, p, p, p };
            __m128d a = perm64( load64( u ), srl64( perm1, pp ) );
            __m128d b = perm64( load64( v ), srl64( perm2, pp ) );
            maskstore64( u, sll64( maskB, pp ), b );
            maskstore64( v, sll64( maskA, pp ), a );

            u = uu;
            v = vv;
            r.x >>= 2;
        }
    }

    // manage elements one at a time

    while( 1 )
    {
        if( flip( &r ) )
        {
            if( v == w )
                break;
            swap( u++, v++ );
        }
        else
        {
            if( u == v )
                break;
            u++;
        }
    }

    // use Fisher-Yates to finish

    while( u < w )
    {
        unsigned int i = random_int( &r, u - t + 1 );
        swap( t + i, u++ );
    }
}
