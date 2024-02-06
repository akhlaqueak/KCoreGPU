#ifndef KCORE_H
#define KCORE_H
#include "common.h"
#define INVALID -1
// #define DEGREESORT
class DeviceGraph{
        
};

class KCore
{
public:
public:
    Index *glBuffer;
    Index *tail;
    Index *prevTail;
    Index *head;
    Index *level;
    Index *count;
    Graph *dp;
    __device__ unsigned int scanIndex(bool pred)
    {
        unsigned int bits = __ballot_sync(FULL, pred);
        unsigned int mask = FULL >> (31 - LANEID);
        unsigned int index = __popc(mask & bits) - pred; // to get exclusive sum subtract pred
        return index;
    }
    __device__ void append(unsigned int v, ui *shBuffer, ui *shTail)
    {
        ui sht = atomicAdd(shTail, 1);
        if (sht < MAX_NV)
        {
            shBuffer[sht] = v;
            return;
        }
        Index t = atomicAdd(tail, 1);
        glBuffer[t] = v;
    }

    __device__ void append(unsigned int v)
    {
        Index t = atomicAdd(tail, 1);
        glBuffer[t] = v;
    }

    __device__ void append(unsigned int v, bool pred)
    {
        unsigned int ind = scanIndex(pred);
        unsigned int loc;
        if (LANEID == 31)
        {
            loc = atomicAdd(tail, ind + pred);
        }
        loc = __shfl_sync(FULL, loc, 31);

        if (pred)
            glBuffer[loc + ind] = v;
    }
    __device__ unsigned int next()
    {
        Index v;
        if (LANEID == 0)
        {
            Index t = atomicAdd(head, 1);
            if (t < prevTail[0])
            {
                // printf("%d-%d-%d-%d \n", t, head[0], tail[0], prevTail[0]);
                v = glBuffer[t];
            }
            else
                v = INVALID;
        }
        v = __shfl_sync(FULL, v, 0);
        return v;
    }

    void allocateMemory(Graph &g)
    {
        chkerr(cudaMallocManaged(&dp, sizeof(Graph)));
        chkerr(cudaMallocManaged(&glBuffer, sizeof(Index) * g.V));
        chkerr(cudaMallocManaged(&tail, sizeof(Index)));
        chkerr(cudaMallocManaged(&prevTail, sizeof(Index)));
        chkerr(cudaMallocManaged(&head, sizeof(Index)));
        chkerr(cudaMallocManaged(&level, sizeof(Index)));
        chkerr(cudaMallocManaged(&count, sizeof(Index)));
        dp->allocateMemory(g);
        tail[0] = 0;
        head[0] = 0;
        prevTail[0] = 0;
        level[0] = 0;
        count[0] = 0;
    }

    __device__ void scan()
    {
        // __shared__ unsigned int stail;
        // __shared__ unsigned int sbuff[1024];
        // __shared__ unsigned int st;

        // if (THID == 0)
        //     stail = 0;

        unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (unsigned int base = 0; base < dp->V; base += N_THREADS)
        {
            unsigned int v = base + global_threadIdx;

            if (v < dp->V && dp->degrees[v] == level[0])
            {
                append(v);
            }

            // bool pred = v < dp->V && dp->degrees[v] == level[0];
            // if(!__ballot_sync(FULL, pred)) continue; // warp found no items to insert
            // append(v, pred);

            // __syncthreads();
            // if (v < dp->V && dp->degrees[v] == level[0])
            // {
            //     unsigned int t = atomicAdd(&stail, 1);
            //     sbuff[t] = v;
            // }
            // __syncthreads();
            // if (THID == 0)
            //     st = atomicAdd(tail, stail);
            // __syncthreads();
            // if (THID < stail)
            //     glBuffer[st + THID] = sbuff[THID];
            // __syncthreads();
            // if (THID == 0)
            //     stail = 0;
        }
    }

    __device__ void loop()
    {
        __shared__ VertexID shBuffer[MAX_NV];
        __shared__ VertexID shTail;
        __shared__ VertexID base;
        if (THID == 0)
        {
            shTail = 0;
            base = 0;
        }
        __syncthreads();

        // while (true)
        // {

        //     VertexID v = next();
        //     if (v == INVALID)
        //         return;
        for (Index wid = head[0] + GLWARPID; wid < prevTail[0]; wid += N_WARPS)
        {
            VertexID v = glBuffer[wid];
            // if(LANEID==0)
            //     printf("%d-%d-%d \n", head[0], tail[0], prevTail[0]);
            examine(v, shBuffer, &shTail);
        }

        while (true)
        {
            __syncthreads(); // syncthreads must be executed by all the threads
            if (base == shTail || base >= MAX_NV)
                break;

            unsigned int i = base + WARPID;
            unsigned int regTail = shTail;
            __syncthreads(); // this call is necessary, so that following update to base is done after everyone get value of

            if (THID == 0)
            {
                base += WARPS_EACH_BLK;
                if (regTail < base)
                    base = regTail;
                // printf("%d ", regTail);
            }

            if (i >= regTail || i >= MAX_NV)
                continue; // this warp won't have to do anything
            // if(!LANEID) printf("i:%d ", i);
            VertexID v = shBuffer[i];
            examine(v, shBuffer, &shTail);
        }

        __syncthreads();
        if(!THID){
            atomicAdd(count, min(shTail, MAX_NV));
        }
    }

    __device__ void examine(unsigned int v, ui *shBuffer, ui *shTail)
    {
        Index start = dp->neighbors_offset[v];
        Index end = dp->neighbors_offset[v + 1];
        // pred=false;

        while (true)
        {
            __syncwarp();
            if (start >= end)
                break;
            unsigned int j = start + LANEID;
            start += 32;
            if (j >= end)
                continue;
            unsigned int u = dp->neighbors[j];
            if (dp->degrees[u] > level[0])
            {
                unsigned int a = atomicSub(dp->degrees + u, 1);
                if (a == level[0] + 1)
                    append(u, shBuffer, shTail);
                if (a <= level[0])
                    // node degree became less than the level after decrementing...
                    atomicAdd(dp->degrees + u, 1);
            }
        }
    }
};

__global__ void scan(KCore kc)
{
    kc.scan();
}

__global__ void loop(KCore kc)
{
    kc.loop();
}

class Degeneracy
{
    Graph g;
    Index *degOrder;
    vector<Index> cShellOffsets;

public:
    Degeneracy(Graph &dg) : g(dg)
    {
        degOrder = new Index[g.V];
        cShellOffsets.push_back(0);
    }

    Index *degenerate()
    {

        KCore kc;
        kc.allocateMemory(g);

        cout << "K-core Computation Started" << endl;

        auto tick = chrono::steady_clock::now();
        while (kc.count[0] + kc.tail[0] < g.V)
        {
            scan<<<BLK_NUMS, BLK_DIM>>>(kc);
            cudaDeviceSynchronize();
            while (kc.prevTail[0] != kc.tail[0])
            {
                kc.head[0] = kc.prevTail[0];
                kc.prevTail[0] = kc.tail[0];
                loop<<<BLK_NUMS, BLK_DIM>>>(kc);
                cudaDeviceSynchronize();
            }
            cout << "*********Completed level: " << kc.level[0] << ", global_count: " << kc.count[0] + kc.tail[0] << " *********" << endl;
            kc.level[0]++;
            cShellOffsets.push_back(kc.tail[0]);
        }
        cout << "Kcore Computation Done" << endl;
        cout << "KMax: " << kc.level[0] - 1 << endl;
        cout << "Kcore-class execution Time: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - tick).count() << endl;
        return kc.glBuffer;
    }

    // void degreeSort()
    // {
    //     ////////////////// degrees sorting after degenracy...
    //     auto tick = chrono::steady_clock::now();

    //     // sort each k-shell vertices based on their degrees.
    //     auto degComp = [&](auto a, auto b)
    //     {
    //         return g.degrees[a] < g.degrees[b];
    //     };

    //     for (int i = 0; i < cShellOffsets.size() - 1; i++)
    //         std::sort(degOrder + cShellOffsets[i], degOrder + cShellOffsets[i + 1], degComp);

    //     VertexID *revOrder = new VertexID[g.V];
    //     // copy back the sorted vertices to rec array...
    //     for (int i = 0; i < g.V; i++)
    //         revOrder[degOrder[i]] = i;
    //     std::swap(degOrder, revOrder);

    //     delete[] revOrder;
    //     cout << "Degree Sorting Time: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - tick).count() << endl;
    // }

    // Graph recode()
    // {
    //     Graph gRec;
    //     gRec.degrees = new unsigned int[g.V];
    //     gRec.neighbors = new unsigned int[g.E];
    //     gRec.neighbors_offset = new unsigned int[g.V + 1];
    //     gRec.V = g.V;
    //     gRec.E = g.E;

    //     auto tick = chrono::steady_clock::now();
    //     cout << "Degrees copied" << endl;
    //     for (int i = 0; i < g.V; i++)
    //     {
    //         gRec.degrees[degOrder[i]] = g.degrees[i];
    //     }

    //     gRec.neighbors_offset[0] = 0;
    //     std::partial_sum(gRec.degrees, gRec.degrees + g.V, gRec.neighbors_offset + 1);

    //     for (int v = 0; v < g.V; v++)
    //     {
    //         unsigned int recv = degOrder[v];
    //         unsigned int start = gRec.neighbors_offset[recv];
    //         unsigned int end = gRec.neighbors_offset[recv + 1];
    //         for (int j = g.neighbors_offset[v], k = start; j < g.neighbors_offset[v + 1]; j++, k++)
    //         {
    //             gRec.neighbors[k] = degOrder[g.neighbors[j]];
    //         }
    //         std::sort(gRec.neighbors + start, gRec.neighbors + end);
    //     }
    //     cout << "Reordering Time: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - tick).count() << endl;

    //     return gRec;
    // }
};
#endif