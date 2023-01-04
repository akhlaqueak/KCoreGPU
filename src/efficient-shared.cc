__global__ void selectNodesAtLevel9(unsigned int *degrees, unsigned int level, unsigned int V, 
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


__global__ void processNodes9(G_pointers d_p, int level, int V, 
                    unsigned int* bufTails, unsigned int* glBuffers, 
                    unsigned int *global_count){
    __shared__ unsigned int shBuffer[MAX_NV], initTail;
    __shared__ unsigned int bufTail;
    __shared__ unsigned int* glBuffer;
    __shared__ unsigned int base;
    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int regTail;
    unsigned int i;
    if(THID==0){
        initTail = bufTail = bufTails[blockIdx.x];
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
        unsigned int v = readFromBuffer(shBuffer, glBuffer, initTail, i);

        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];
        bool pred = false;
        unsigned int u;

        while(true){
            unsigned int loc = scanIndexHellis(pred, &bufTail);
            if(pred){
                writeToBuffer(shBuffer, glBuffer, initTail, loc, u);
            }
            if(start >= end) break;
            pred = false;
            unsigned int j = start + lane_id;
            start += WARP_SIZE;
            if(j >= end) continue;

            u = d_p.neighbors[j];
            if( d_p.degrees[u] > level){
                
                unsigned int a = atomicSub(d_p.degrees+u, 1);
                pred = (a == level+1);


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


int kcoreSharedMemEfficient(Graph &data_graph){

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

        selectNodesAtLevel9<<<BLK_NUMS, BLK_DIM>>>(data_pointers.degrees, level, 
                        data_graph.V, bufTails, glBuffers);

        processNodes9<<<BLK_NUMS, BLK_DIM>>>(data_pointers, level, data_graph.V, 
                        bufTails, glBuffers, global_count);

        chkerr(cudaMemcpy(&count, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));    
        // cout<<"*********Completed level: "<<level<<", global_count: "<<count<<" *********"<<endl;
        level++;
    }
	// cout<<"Done"<<endl;

    auto end = chrono::steady_clock::now();
    
    
    // cout << "Elapsed Time: "
    // << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
    // cout <<"MaxK: "<<level-1<<endl;
    
    
	// get_results_from_gpu(data_graph, data_pointers);
    
    cudaFree(glBuffers);
    free_graph_gpu_memory(data_pointers);
    // if(write_to_disk){
    //     cout<<"Writing kcore to disk started... "<<endl;
    //     data_graph.writeKCoreToDisk(data_file);
    //     cout<<"Writing kcore to disk completed... "<<endl;
    // }

    return chrono::duration_cast<chrono::milliseconds>(end - start).count();

}
