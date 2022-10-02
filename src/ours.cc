
__global__ void selectNodesAtLevel1(unsigned int *degrees, unsigned int level, unsigned int V, 
                 unsigned int* bufTails, unsigned int* glBuffers){

    __shared__ unsigned int* glBuffer; 
    __shared__ unsigned int bufTail; 
    
    if(THID == 0){
        bufTail = 0;
        glBuffer = glBuffers + blockIdx.x*GLBUFFER_SIZE;
    }
    __syncthreads();

    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + global_threadIdx; 

        if(v >= V) continue;

        if(degrees[v] == level){
            unsigned int loc = atomicAdd(&bufTail, 1);
            writeToBuffer(glBuffer, loc, v);
        }
    }

    __syncthreads();

    if(THID == 0) 
    {
        bufTails [blockIdx.x] = bufTail;
    }
}




__global__ void processNodes1(G_pointers d_p, int level, int V, 
                    unsigned int* bufTails, unsigned int* glBuffers, 
                    unsigned int *global_count){
    __shared__ unsigned int bufTail;
    __shared__ unsigned int* glBuffer;
    __shared__ unsigned int base;
    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int regTail;
    unsigned int i;
    if(THID==0){
        bufTail = bufTails[blockIdx.x];
        base = 0;
        glBuffer = glBuffers + blockIdx.x*GLBUFFER_SIZE; 
        assert(glBuffer!=NULL);
    }

    // bufTail is being incrmented within the loop, 
    // warps should process all the nodes added during the execution of loop
    
    // for(unsigned int i = warp_id; i<bufTail ; i +=warps_each_block ){
    // this for loop is a wrong choice, as many threads will exit from the loop checking the condition
    while(true){
        __syncthreads(); //syncthreads must be executed by all the threads
        if(base == bufTail) break; // all the threads will evaluate to true at same iteration
        i = base + warp_id;
        regTail = bufTail;
        __syncthreads();

        if(i >= regTail) continue; // this warp won't have to do anything            

        if(THID == 0){
            // base += min(WARPS_EACH_BLK, regTail-base)
            // update base for next iteration
            base += WARPS_EACH_BLK;
            if(regTail < base )
                base = regTail;
        }
        //bufTail is incremented in the code below:
        unsigned int v = readFromBuffer(glBuffer, i);
        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];


        while(true){
            __syncwarp();

            if(start >= end) break;

            unsigned int j = start + lane_id;
            start += WARP_SIZE;
            if(j >= end) continue;

            unsigned int u = d_p.neighbors[j];
            if(*(d_p.degrees+u) > level){
                
                unsigned int a = atomicSub(d_p.degrees+u, 1);
            
                if(a == level+1){
                    unsigned int loc = atomicAdd(&bufTail, 1);

                        writeToBuffer(glBuffer, loc, u);
                }

                if(a <= level){
                    // node degree became less than the level after decrementing... 
                    atomicAdd(d_p.degrees+u, 1);
                }
            }
        }

    }

    if(THID == 0 && bufTail>0){
        atomicAdd(global_count, bufTail); // atomic since contention among blocks
    }
}


int kcore(Graph &data_graph){

    G_pointers data_pointers;

    malloc_graph_gpu_memory(data_graph, data_pointers);

    unsigned int level = 0;
    unsigned int count = 0;
    unsigned int* global_count  = NULL;
    unsigned int* bufTails  = NULL;
    unsigned int* glBuffers     = NULL;

    chkerr(cudaMalloc(&global_count, sizeof(unsigned int)));
    chkerr(cudaMalloc(&bufTails, sizeof(unsigned int)*BLK_NUMS));
    cudaMemset(global_count, 0, sizeof(unsigned int));
    chkerr(cudaMalloc(&glBuffers,sizeof(unsigned int)*BLK_NUMS*GLBUFFER_SIZE));
       
    
	// cout<<"K-core Computation Started";

    auto start = chrono::steady_clock::now();
    while(count < data_graph.V){
        cudaMemset(bufTails, 0, sizeof(unsigned int)*BLK_NUMS);

        selectNodesAtLevel1<<<BLK_NUMS, BLK_DIM>>>(data_pointers.degrees, level, 
                        data_graph.V, bufTails, glBuffers);

        processNodes1<<<BLK_NUMS, BLK_DIM>>>(data_pointers, level, data_graph.V, 
                        bufTails, glBuffers, global_count);

        chkerr(cudaMemcpy(&count, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));    
        // cout<<"*********Completed level: "<<level<<", global_count: "<<count<<" *********"<<endl;
        level++;
    }
    data_graph.kmax = level-1;
    auto end = chrono::steady_clock::now();

    cudaFree(glBuffers);
    free_graph_gpu_memory(data_pointers);


    return chrono::duration_cast<chrono::milliseconds>(end - start).count();

}
