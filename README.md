# KCoreGPU
The k-core of a graph is the largest induced
subgraph with minimum degree k. The problem of k-core
decomposition finds the k-cores of a graph for all valid values
of k. In this work we developed a highly
optimized peeling algorithm on a GPU for k-core decomposition. 

## Compilation and Execution
    
    $ cd /KCoreGPU/
    $ make 
    $ ./kcore graph.txt
    By default, nvcc and g++ compiler is used.
    
    
## Sample Datasets

    Some datasets are provided in dataset folder, please use them to experiment the execution of our algorithm. e.g. 
    $ ./kcore dataset/amazon0601.txt
    
## Citing KCoreGPU

    If you use KCoreGPU, please cite our paper:
    TBD... 
