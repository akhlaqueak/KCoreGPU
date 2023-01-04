#include "../inc/common.h"

enum{INCLUSIVE, EXCLUSIVE};
__device__ unsigned int scanWarpHellis(volatile unsigned int* addresses, unsigned int type){
    
    const unsigned int lane_id = THID & 31;
    for(int i=1; i<WARP_SIZE; i*=2){
        if(lane_id >= i)
            addresses[THID] += addresses[THID-i];
        // __syncwarp();
    }

    
    if(type == INCLUSIVE)
        return addresses[THID];
    else{
        return (lane_id>0)? addresses[THID-1]:0;
    }    
}

// returns index to write after scanning a warp
__device__ unsigned int scanIndexBallot(bool pred, unsigned int* bufTail)
{
    unsigned int laneid = THID%32;
    unsigned int bits = __ballot_sync(FULL, pred);
    unsigned int mask = FULL >> (31 - laneid);
    unsigned int index = __popc(mask & bits) - pred; // to get exclusive sum subtract pred
    unsigned int btail = 0;
    if(laneid==31){
        btail = atomicAdd(bufTail, index+pred);
    }
    btail = __shfl_sync(FULL, btail, 31);
    index+=btail;
    return index;
}


__device__ unsigned int scanIndexHellis(bool pred, unsigned int* bufTail)
{
    const unsigned int laneid = THID & 31;
    __shared__ volatile unsigned int addresses[BLK_DIM];
    addresses[THID] = pred; 
    for(int i=1; i<WARP_SIZE; i*=2){
        if(laneid >= i)
            addresses[THID] += addresses[THID-i];
        // __syncwarp();
    }
    unsigned int btail;
    if(laneid==31)
        btail = atomicAdd(bufTail, addresses[THID]);
    btail = __shfl_sync(FULL, btail, 31);

    return addresses[THID]+btail-pred; // exclusive scan requires -pred
}


__device__ unsigned int scanWarpBallot(volatile unsigned int* addresses, unsigned int type){
    uint lane_id = THID & 31;
    uint bits = __ballot_sync(0xffffffff, addresses[THID]);
    uint mask = 0xffffffff >> (31-lane_id);
    addresses[THID] = __popc(mask & bits);
    if(type == INCLUSIVE)
        return addresses[THID];
    else
        return lane_id>0? addresses[THID-1] : 0;
}



__device__ void scanBlock(volatile unsigned int* addresses, unsigned int type){
    const unsigned int lane_id = THID & 31;
    const unsigned int warp_id = THID >> 5;
    
    // unsigned int val = scanWarpBallot(addresses, type);
    unsigned int val = scanWarpHellis(addresses, type);
    __syncthreads();

    if(lane_id==31)
        addresses[warp_id] = addresses[THID];
    __syncthreads();

    if(warp_id==0)
    // can't use ballot scan here... 
        scanWarpHellis(addresses, INCLUSIVE);
    __syncthreads();

    if(warp_id>0)
        val += addresses[warp_id-1];
    __syncthreads();

    addresses[THID] = val;
    __syncthreads();
    
}




__device__ void compactWarpHellis(bool* predicate, volatile unsigned int* addresses, unsigned int* temp, 
        unsigned int* glBuffer, unsigned int* bufTail){
    
    const unsigned int lane_id = THID & 31;
    addresses[THID] = predicate[THID];
    unsigned int address = scanWarpHellis(addresses, EXCLUSIVE);
    // unsigned int address = scanWarpBallot(addresses, EXCLUSIVE);
    unsigned int bTail;
    
    if(lane_id==WARP_SIZE-1){
        bTail = atomicAdd(bufTail, address + predicate[THID]);
    }
    bTail = __shfl_sync(0xFFFFFFFF, bTail, WARP_SIZE-1);

    address += bTail;
    if(predicate[THID])
        writeToBuffer(glBuffer, address, temp[THID]);
    predicate[THID] = 0;
}
__device__ void compactWarpBallot(bool* predicate, volatile unsigned int* addresses, unsigned int* temp, 
        unsigned int* glBuffer, unsigned int* bufTail){
    
    const unsigned int lane_id = THID & 31;
    addresses[THID] = predicate[THID];
    // unsigned int address = scanWarpHellis(addresses, EXCLUSIVE);
    unsigned int address = scanWarpBallot(addresses, EXCLUSIVE);
    unsigned int bTail;
    
    if(lane_id==WARP_SIZE-1){
        bTail = atomicAdd(bufTail, address + predicate[THID]);
    }
    bTail = __shfl_sync(0xFFFFFFFF, bTail, WARP_SIZE-1);

    address += bTail;
    if(predicate[THID])
        writeToBuffer(glBuffer, address, temp[THID]);
    predicate[THID] = 0;
}
__device__ void compactBlock(bool* predicate, volatile unsigned int* addresses, unsigned int* temp, 
        unsigned int* glBuffer, unsigned int* bufTail){
    __shared__ unsigned int bTail;
    
    addresses[THID] = predicate[THID];
    scanBlock(addresses, EXCLUSIVE);
    if(THID == BLK_DIM-1){
        bTail = atomicAdd(bufTail, addresses[THID] + predicate[THID]);
    }
    __syncthreads();

    if(predicate[THID])
        writeToBuffer(glBuffer, addresses[THID] + bTail, temp[THID]);
    predicate[THID] = 0;
}