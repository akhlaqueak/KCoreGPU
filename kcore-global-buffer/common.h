
#ifndef CUTS_COMMON_H
#define CUTS_COMMON_H
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <bits/stdc++.h>
#include "omp.h"

#define BLK_NUMS 56
#define BLK_DIM 1024
#define WARPS_EACH_BLK (BLK_DIM >> 5)
#define MAX_NV 12000
#define N_THREADS (BLK_DIM * BLK_NUMS)
#define N_WARPS (BLK_NUMS * WARPS_EACH_BLK)
#define GLBUFFER_SIZE 1000000
#define THID threadIdx.x
#define WARP_SIZE 32
#define UINT unsigned int
#define DS_LOC string("")
#define OUTPUT_LOC string("../output/")
#define REP 1
#define WARPID (THID >> 5)
#define LANEID (THID & 31)
#define BLKID blockIdx.x
#define FULL 0xFFFFFFFF
#define GLWARPID (BLKID * WARPS_EACH_BLK + WARPID)
#define GTHID (BLKID*N_THREADS+THID)
// #define GLWARPID (blockIdx.x*(BLK_DIM/32)+(threadIdx.x>>5))

// ****************BK specific parameters********************
// CHUNK is how many subproblems processes initially from the degeneracy ordered pool
// chunk value is reduced to half everytime a memory flow occurs. 
// It can't reduce once reached to MINCHUNK
#define MAXCHUNK 1'000'000
#define MINCHUNK 5
#define MINSTEP MINCHUNK

#define HOSTCHUNK 1'000'000 // chunk size for host memory buffer

// NSUBS is size of sg.offsets array, since every subgraph requires 2 items,
// hence number of supported subgraphs are half of this number
// #define NSUBS 8e8
// BUFFSIZE is length of sg.vertices and sg.labels
// #define BUFFSIZE 8e8
// tempsize is max size of a subgraph stored in temp area, This size is per warp
// in general it should be the size of max degree of supported graph
#define TEMPSIZE 100'000

#define HOST_BUFF_SZ 4'000'000'000

// Reduction and Increment factors
#define DECFACTOR 8
#define INCFACTOR 2
#define THRESH 0
#define ADDPC   0.15
#define SUCCESS_ITER 4

#define R 'r'
#define P 'p'
#define X 'x'
#define Q 'q'

#define DEV __device__
#define DEVHOST __device__ __host__

typedef unsigned int Index ;
typedef unsigned int ui ;
typedef  unsigned int VertexID;
typedef  char Label;

// #define DEGREESORT


using namespace std;
inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout << cudaGetErrorString(code) << std::endl;
        exit(-1);
    }
}


#endif // CUTS_COMMON_H
