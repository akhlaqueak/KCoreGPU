

#ifndef CUTS_GPU_MEMORY_ALLOCATION_H
#define CUTS_GPU_MEMORY_ALLOCATION_H
#include "./graph.h"
void malloc_graph_gpu_memory(Graph &g,G_pointers &p);
void get_results_from_gpu(Graph &g,G_pointers &p);
void free_graph_gpu_memory(G_pointers &p);
inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout<<cudaGetErrorString(code)<<std::endl;
        exit(-1);
    }
}
#endif //CUTS_GPU_MEMORY_ALLOCATION_H
