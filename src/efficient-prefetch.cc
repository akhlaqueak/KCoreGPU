

__global__ void selectNodesAtLevel7(unsigned int *degrees, unsigned int level, unsigned int V, 
                 unsigned int* bufTails, unsigned int* glBuffers){

    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int temp[BLK_DIM];
    __shared__ volatile unsigned int addresses[BLK_DIM];
    __shared__ unsigned int bufTail;
    __shared__ unsigned int* glBuffer;
    if(THID==0){
        bufTail = 0;
        glBuffer = glBuffers+(blockIdx.x*GLBUFFER_SIZE);
    }

    unsigned int glThreadIdx = blockIdx.x * BLK_DIM + THID; 

    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + glThreadIdx; 

        // all threads should get some value, if vertices are less than n_threads, rest of the threads get zero
        predicate[THID] = (v<V)? (degrees[v] == level) : 0;
        if(predicate[THID]) temp[THID] = v;

        compactBlock(predicate, addresses, temp, glBuffer, &bufTail);        
        
        __syncthreads();
            
    }

    if(THID==0){
        bufTails[blockIdx.x] = bufTail;
    }
}







__global__ void processNodes7(G_pointers d_p, int level, int V, 
                    unsigned int* bufTails, unsigned int* glBuffers, 
                    unsigned int *global_count){
                          
    __shared__ volatile unsigned int addresses[BLK_DIM];
    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int temp[BLK_DIM];
    __shared__ unsigned int bufTail;
    __shared__ unsigned int base;
    __shared__ unsigned int* glBuffer;
    __shared__ unsigned int prefv[32];
    __shared__ int npref;

    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int start, end, regnpref, regTail, v;
    
    if(THID==0){
        bufTail = bufTails[blockIdx.x];
        base = 0;
        npref = 0;
        glBuffer = glBuffers + blockIdx.x * GLBUFFER_SIZE; 
    }

    __syncthreads();
    
    if(warp_id > 0)
        if(warp_id-1<bufTail){
            prefv[warp_id] = readFromBuffer(glBuffer, warp_id-1);
        }
    if(THID==0){
        npref = min(WARPS_EACH_BLK-1, bufTail-base);
    }
    
    predicate[THID] = 0;
    // bufTail is being incrmented within the loop, 
    // warps should process all the nodes added during the execution of loop
    // for that purpose e_processes is introduced, is incremented whenever a warp takes a job. 
    
    
    // for(unsigned int i = warp_id; i<bufTail ; i = warp_id + base){
    // this for loop is a wrong choice, as many threads will exit from the loop checking the condition
    while(true){
        __syncthreads(); //syncthreads must be executed by all the threads
        if(warp_id <= npref){
            v = prefv[warp_id];
        }
        if(base == bufTail) break; // all the threads will evaluate to true at same iteration
        regnpref = npref;
        regTail = bufTail;
        __syncthreads();
        if(warp_id > regnpref) continue; 

        if(warp_id == 0){
            if(lane_id == 0){
                // update base for next iteration
                base += npref;
            } 
            __syncwarp(); // so that other lanes can see updated base value
            
            if(lane_id > 0){
                int j = base + lane_id - 1;
                npref = min(WARPS_EACH_BLK-1, regTail-base);
                if(j < regTail){
                    prefv[lane_id] = readFromBuffer(glBuffer, j);
                }
            }
            continue; // warp0 doesn't process nodes. 
        }

        start = d_p.neighbors_offset[v];
        end = d_p.neighbors_offset[v+1];
        while(true){
            // __syncwarp();

            compactWarpHellis(predicate, addresses, temp, glBuffer, &bufTail);
            if(start >= end) break;

            unsigned int j = start + lane_id;
            start += WARP_SIZE;
            if(j >= end) continue;

            unsigned int u = d_p.neighbors[j];
            if(*(d_p.degrees+u) > level){
                unsigned int a = atomicSub(d_p.degrees+u, 1);
            
                if(a == level+1){
                    temp[THID] = u;
                    predicate[THID] = 1;
                    // unsigned int loc = atomicAdd(&bufTail, 1);
                    // writeToBuffer(shBuffer, glBuffer, loc, u);
                }

                if(a <= level){
                    // node degree became less than the level after decrementing... 
                    atomicAdd(d_p.degrees+u, 1);
                }
            }
        }

    }
    __syncthreads();

    if(THID == 0 ){
        if(bufTail>0) atomicAdd(global_count, bufTail); // atomic since contention among blocks
        // if(glBuffer!=NULL) free(glBuffer);
    }

}



int kcoreEfficientScanPrefetch(Graph &data_graph){

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

        selectNodesAtLevel7<<<BLK_NUMS, BLK_DIM>>>(data_pointers.degrees, level, 
                        data_graph.V, bufTails, glBuffers);

        processNodes7<<<BLK_NUMS, BLK_DIM>>>(data_pointers, level, data_graph.V, 
                        bufTails, glBuffers, global_count);

        chkerr(cudaMemcpy(&count, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));    
        // cout<<"*********Completed level: "<<level<<", global_count: "<<count<<" *********"<<endl;
        level++;
    }
	// cout<<"Done "<<"Kmax: "<<level-1<<endl;

    auto end = chrono::steady_clock::now();

    cudaFree(glBuffers);
    free_graph_gpu_memory(data_pointers);


    return chrono::duration_cast<chrono::milliseconds>(end - start).count();

}
