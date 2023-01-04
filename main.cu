#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>


#include "./inc/gpu_memory_allocation.h"
#include "./src/buffer.cc"
#include "./src/scans.cc"
#include "./src/ours.cc"
#include "./src/ours-shared.cc"
#include "./src/ours-prefetch.cc"
#include "./src/efficient.cc"
#include "./src/ballot-prefetch.cc"
#include "./src/ballot.cc"
#include "./src/efficient-prefetch.cc"
#include "./src/ballot-shared.cc"
#include "./src/efficient-shared.cc"


template<class T>
void repSimulation(int (*kern)(T), Graph& g){
    float sum=0;
    int rep = 1; // number of iterations... 
    for(int i=0;i<rep;i++){
        unsigned int t = (*kern)(g);
        cout<<t<<" ";
        sum+=t;
    }
    cout<<endl;
}

void STDdegrees(Graph& g){
    double sum = std::accumulate(g.degrees, g.degrees+g.V, 0.0);
    double mean = sum / g.V;

    std::vector<double> diff(g.V);
    std::transform(g.degrees, g.degrees+g.V, diff.begin(),
                std::bind2nd(std::minus<double>(), mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / g.V);
    cout<<stdev<<endl;
}   

int main(int argc, char *argv[]){
    if (argc < 2) {
        cout<<"Please provide data file"<<endl;
        exit(-1);
    }
    std::string ds = argv[1];

    cout<<"Graph loading Started... "<<endl;    
    Graph g(ds);
    cout<<"******************  "<<ds<<" ****************** "<<endl;
    cout<<"V: "<< g.V<<endl;
    cout<<"E: "<< g.E<<endl;
    
    // cout<<"******************  "<<ds<<" ****************** ";
    // STDdegrees(g);

    cout<<"Ours: ";
    repSimulation(kcore, g);
    cout<<"Kmax: "<<g.kmax<<endl;

    cout<<"SM:  ";
    repSimulation(kcoreSharedMem, g);

    
    cout<<"VP: ";
    repSimulation(kcorePrefetch, g);

    cout<<"BC: ";
    repSimulation(kcoreBallotScan, g);

    cout<<"BC + SM: ";
    repSimulation(kcoreSharedMemBallot, g);

    cout<<"BC + VP: ";
    repSimulation(kcoreBallotScanPrefetch, g);


    cout<<"EC: ";
    repSimulation(kcoreEfficientScan, g);    
    
    cout<<"EC + SM: ";
    repSimulation(kcoreSharedMemEfficient, g);

    cout<<"EC + VP: ";
    repSimulation(kcoreEfficientScanPrefetch, g);

    return 0;
}
