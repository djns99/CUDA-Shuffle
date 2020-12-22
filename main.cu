#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "ThrustInclude.h"

uint64_t fact( uint64_t n )
{
    uint64_t res = 1;
    for( uint64_t i = 2; i <= n; i++ )
        res *= n;
    return n;
}
#include <random>

int main( int argc, char** argv )
{
    thrust::uniform_int_distribution<uint64_t> dist;
    // thrust::default_random_engine engine{ 0 };
    // thrust::taus88 engine{ 0xdeadbeef };
    thrust::ranlux48 engine{ 0 };
    for( uint64_t i = 0; i < 1e4; i++ )
        std::cout << std::hex << dist( engine ) << std::endl;

    /*
    RaoSandeliusShuffle<> shuffle;
    DefaultRandomGenerator gen;
    const uint64_t count = 5;
    std::unordered_map<std::string, uint64_t> map;
    int iters = 1e6;
    for( int i = 0; i < iters; i++ )
    {
        std::vector<uint64_t> in_nums( count );
        std::vector<uint64_t> out_nums( count );
        std::iota( in_nums.begin(), in_nums.end(), 0 );
        shuffle( in_nums, out_nums, gen() );

        std::stringstream ss;
        for( uint64_t num : out_nums )
            ss << num << ", ";
        map[ss.str()]++;

        if( ( i % 10000 ) == 0 )
            std::cout << i << std::endl;
    }

    for( auto& pair : map )
    {
        std::cout << pair.first << " occurred " << pair.second << " ("
                  << pair.second / (double)iters << ")" << std::endl;
    }
    std::cout << "Expected: " << iters / fact( count ) << std::endl;

    return 0;
    */
}