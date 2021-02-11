#pragma once
#include "DefaultRandomGenerator.h"
#include "shuffle/Shuffle.h"
#include <algorithm>
#include <thread>
#include <vector>

template <class ContainerType = std::vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class RaoSandeliusShuffle : public Shuffle<ContainerType, RandomGenerator>
{
private:
    unsigned long cutoff = 0x20000;
    unsigned long cutoff2 = 0x20000;

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


    template <class Iterator>
    void raoSandeliusShuffle( Iterator begin, Iterator end, RandomGenerator& g )
    {
        if( end - begin < cutoff )
        {
            std::shuffle( begin, end, g );
            return;
        }
        Iterator i = begin;
        Iterator j = end;
        Flipper flip( g );
        while( i != j )
        {
            if( flip() )
                std::swap( *i, *( --j ) );
            else
                i++;
        }

        if( end - begin < cutoff2 )
        {
            raoSandeliusShuffle( begin, i, g );
            raoSandeliusShuffle( i, end, g );
        }
        else
        {
            RandomGenerator local_g( g );
            std::thread t( [=, &local_g]() {
                raoSandeliusShuffle( begin, i, local_g );
            } );
            raoSandeliusShuffle( i, end, g );
            t.join();
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
        raoSandeliusShuffle( out_container.begin(), out_container.begin() + num, g );
    }
};