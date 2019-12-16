#pragma once
#include <cuda_runtime.h>

#define checkCudaError( ans )                           \
    {                                                   \
        assertCudaError( ( ans ), __FILE__, __LINE__ ); \
    }
inline void assertCudaError( cudaError_t code, std::string file, int line )
{
    if( code != cudaSuccess )
    {
        throw std::runtime_error( "CUDA Error " + std::string( cudaGetErrorString( code ) ) + " " +
                                  file + ":" + std::to_string( line ) );
    }
}