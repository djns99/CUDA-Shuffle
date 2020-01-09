#include "random.h"
#include <immintrin.h>
#include <string.h>

#define load32( x ) _mm_loadu_ps( (float*)( x ) )
#define perm32( x, y ) _mm_permutevar_ps( ( __m128 )( x ), ( __m128i )( y ) )
#define maskstore32( x, y, z ) _mm_maskstore_ps( (float*)( x ), ( __m128i )( y ), ( __m128 )( z ) )
#define sll32( x, y ) _mm_sll_epi32( ( __m128i )( x ), ( __m128i )( y ) )
#define srl32( x, y ) _mm_srl_epi32( ( __m128i )( x ), ( __m128i )( y ) )

#define load64( x ) _mm128_loadu_ps( (float*)( x ) )
#define perm64( x, y ) _mm128_permutevar_ps( ( __m256 )( x ), ( __m256i )( y ) )
#define maskstore64( x, y, z ) _mm128_maskstore_pd( (double*)( x ), ( __m256i )( y ), ( __m256 )( z ) )
#define sll64( x, y ) _mm128_sll_epi64( ( __m256i )( x ), ( __m256i )( y ) )
#define srl64( x, y ) _mm128_srl_epi64( ( __m256i )( x ), ( __m256i )( y ) )

#define popcnt _mm_popcnt_u64

template<class T>
T* split( T* start, T* end )
{
    T* mid = start;
    struct random r = { 0, 0 };

    if constexpr( sizeof(T) == 4)
    {
        __v4si perm1 = { 303239696, 1803315264, 3166732288, 3221225472 };
        __v4si perm2 = { 0, 1077952576, 2483065856, 3918790656 };
        __v4si mask1 = { 1073741823, 54476799, 197439, 3 };
        __v4si mask2 = { 858993459, 252645135, 16711935, 65535 };

        while( start <= end - 4 )
        {
            int p = random_bits( &r, 4 );

            __v4si pp = { 2 * p, 2 * p, 2 * p, 2 * p };
            __m128 a = perm32( load32( start ), srl32( perm1, pp ) );
            __m128 b = perm32( load32( mid ), srl32( perm2, pp ) );
            maskstore32( start, sll32( mask2, pp ), b );
            maskstore32( mid, sll32( mask1, pp ), a );
            start += 4;
            mid += popcnt( p );
        }
    } else {
        __v2di perm1 = { 0x4, 0x8 };
        __v2di perm2 = { 0x0, 0x8 };

        __v2di maskA = { 0x7FFFFFFFFFFFFFFF, 0x1FFFFFFFFFFFFFFF };
        __v2di maskB = { 0x5FFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF };

        while( start <= end - 4 )
        {
            int p = random_bits( &r, 2 );

            __v4si pp = { p, p, p, p };
            __m128 a = perm64( load64( start ), srl64( perm1, pp ) );
            __m128 b = perm64( load64( mid ), srl64( perm2, pp ) );

            maskstore64( start, sll64( maskB, pp ), b );
            maskstore64( mid, sll64( maskA, pp ), a );
            start += 4;
            mid += popcnt( p );
        }
    }

    while( start < end )
    {
        if( flip( &r ) )
            swap( start, mid++ );
        start++;
    }

    return mid;
}
