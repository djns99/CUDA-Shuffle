#pragma once
#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"
#include <algorithm>
#include <thread>
#include <vector>

template <class ContainerType = std::vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class MergeShuffle : public Shuffle<ContainerType, RandomGenerator>
{
private:
    static constexpr unsigned long cutoff = 0x10000;
    std::vector<DefaultRandomGenerator> generators;

    struct Flipper
    {
        Flipper( RandomGenerator& generator ) : g( generator )
        {
        }
        RandomGenerator& g;
        uint64_t current = 0;
        uint64_t index = 0;
        bool operator()()
        {
            if( index == 0 )
                current = g();
            bool res = ( current >> index ) & 1;
            index = ( index + 1 ) % 64;
            return res;
        }
    };

    static inline unsigned long randomInt( Flipper flip, unsigned long n )
    {
        unsigned long v = 1;
        unsigned long d = 0;
        while( true )
        {
            d += d + flip();
            v += v;

            if( v >= n )
            {
                if( d < n )
                    return d;
                v -= n;
                d -= n;
            }
        }
    }

    template <class T>
    void merge( T* start, uint64_t mid_idx, uint64_t end_idx, RandomGenerator& g )
    {
        T* const original_start = start;
        T* mid = start + mid_idx;
        T* end = start + end_idx;
        Flipper flip( g );
        while( true )
        {
            if( flip() )
            {
                if( start == mid )
                    break;
            }
            else
            {
                if( mid == end )
                    break;
                std::swap( *start, *mid );
                mid++;
            }
            start++;
        }

        while( start != end )
        {
            const uint64_t num_processed = start - original_start;
            const uint64_t index = randomInt( flip, num_processed );
            std::swap( *( original_start + index ), *start );
            start++;
        }
    }

    template <class T>
    void mergeShuffle( T* t, uint64_t n, RandomGenerator& g )
    {
        // Calculate the number of divisions to reach the cutoff
        unsigned int c = 0;
        while( ( n >> c ) > cutoff )
            c++;
        unsigned int q = 1 << c;
        unsigned long nn = n;

        if( generators.capacity() < q )
            generators.reserve( q );
        while( generators.size() < q )
            generators.emplace_back( g() );

            // Launch thread for local fisher yates
#pragma omp parallel for
        for( unsigned int i = 0; i < q; i++ )
        {
            unsigned long j = nn * i >> c;
            unsigned long k = std::min( nn * ( i + 1 ) >> c, nn );
            assert( j < nn );
            assert( k <= nn );
            std::shuffle( t + j, t + k, this->generators[i] );
        }

        for( unsigned int p = 1; p < q; p += p )
        {
#pragma omp parallel for
            for( unsigned int i = 0; i < q; i += 2 * p )
            {
                unsigned long j = nn * i >> c;
                unsigned long k = nn * ( i + p ) >> c;
                unsigned long l = std::min( nn * ( i + 2 * p ) >> c, nn );
                assert( j < nn );
                assert( k < nn );
                assert( l <= nn );
                merge( t + j, k - j, l - j, this->generators[i] );
            }
        }
    }

public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        if( &in_container != &out_container )
        {
            // Copy if we are not doing an inplace operation
            std::copy( in_container.begin(), in_container.begin() + num, out_container.begin() );
        }

        RandomGenerator g( seed );
        mergeShuffle( out_container.data(), num, g );
    }

    bool isDeterministic() const override
    {
        return false;
    }
};