#include <iostream>
#include "shuffle/CzumajShuffle.h"
#include <sstream>
#include <string>
#include <unordered_map>

//#include "ThrustInclude.h"

uint64_t fact( uint64_t n )
{
    uint64_t res = 1;
    for( uint64_t i = 2; i <= n; i++ )
        res *= n;
    return n;
}


int main( int argc, char** argv )
{
    CzumajBijection b;
    DefaultRandomGenerator g( 1 );
    const uint64_t capacity = 100;
    b.init(capacity, g);
    for( uint64_t i = 0; i < capacity; i++)
        std::cout << i << ":" << b( i ) << std::endl;


    /*
    PeriodicLCGBijectiveShuffle<> shuffle;
    DefaultRandomGenerator gen;
    const uint64_t count = 5;
    std::unordered_map<std::string, uint64_t> map;
    int iters = 1e6;
    for( int i = 0; i < iters; i++ )
    {
        thrust::device_vector<uint64_t> in_nums( count );
        thrust::device_vector<uint64_t> out_nums( count );
        thrust::sequence( in_nums.begin(), in_nums.end(), 0 );
        shuffle( in_nums, out_nums, gen() );

        thrust::host_vector<uint64_t> host( out_nums.begin(), out_nums.end() );
        std::stringstream ss;
        for( uint64_t num : host )
            ss << num << ", ";
        map[ss.str()]++;

        if( ( i % 10000 ) == 0 )
            std::cout << i << std::endl;
    }

    for( auto& pair : map )
    {
        std::cout << pair.first << " occurred " << pair.second << " (" << pair.second / (double)iters << ")" << std::endl;
    }
    std::cout << "Expected: " << iters / fact(count) << std::endl;

    return 0;
     */
}