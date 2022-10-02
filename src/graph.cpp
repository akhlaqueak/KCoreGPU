
#include "../inc/graph.h"
bool Graph::readSerialized(string input_file){
    ifstream file;
    file.open(string(OUTPUT_LOC) + string("serialized-") + input_file);
    if(file){
        cout<<"Reading serialized file... "<<endl;
        file>>V;
        file>>E;
        degrees = new unsigned int[V];
        neighbors_offset = new unsigned int[V+1];
        neighbors = new unsigned int[E];
        for(int i=0;i<V;i++)
            file>>degrees[i];
        for(int i=0;i<V+1;i++)
            file>>neighbors_offset[i];
        for(int i=0;i<E;i++)
            file>>neighbors[i];
        file.close();
        dmax = *max_element(degrees, degrees+V);
        return true;
    }else{
        cout<<"readSerialized: File couldn't open"<<endl;
    }

    return false;
}

void Graph::writeSerialized(string input_file){

    ofstream file;
    file.open(string(OUTPUT_LOC) + string("serialized-") + input_file);
    if(file){
        file<<V<<endl;
        file<<E<<endl;
        for(int i=0;i<V;i++)
            file<<degrees[i]<<endl;
        for(int i=0;i<V+1;i++)
            file<<neighbors_offset[i]<<' ';
        for(int i=0;i<E;i++)
            file<<neighbors[i]<<' ';
        file.close();
    }
    else{
        cout<<"writeSerialized: File couldn't open"<<endl;
    }
}

void Graph::readFile(string input_file){

    double load_start = omp_get_wtime();
    ifstream infile;
    infile.open(input_file);
    if(!infile){
        cout<<"load graph file failed "<<endl;
        exit(-1);
    }


    unsigned int s, t;

/**
 * @brief Dataset format:
 * # Number of nodes
 * source destination
 * source destination
 * source destination
 * source destination
 * 
 */
    char ch;
    infile>>ch; // just to eat #
    infile>>V; // read number of nodes... 


    vector<set<unsigned int>> ns(V);

    while(infile>>s>>t){
        if(s==t) continue; // remove self loops
        ns[s].insert(t);
        ns[t].insert(s);
    }

    infile.close();
    
    degrees = new unsigned int[V];

    #pragma omp parallel for
    for(int i=0;i<V;++i){
        degrees[i] = ns[i].size();
    }

    neighbors_offset = new unsigned int[V+1];
    neighbors_offset[0] = 0;
    partial_sum(degrees, degrees+V, neighbors_offset+1);

    E = neighbors_offset[V];
    neighbors = new unsigned int[E];

    #pragma omp parallel for
    for(int i=0;i<V;i++){
        auto it = ns[i].begin();
        for(int j=neighbors_offset[i]; j < neighbors_offset[i+1]; j++, it++)
            neighbors[j] = *it;
    }
}

void Graph::writeKCoreToDisk(std::string file){
    // writing kcore in json dictionary format
    std::ofstream out(OUTPUT_LOC + string("pkc-kcore-") + file);

    out<<"{ ";
   
    for(unsigned long long int i=0;i<V;++i)
            // not writing zero degree nodes, because certain nodes in dataset are not present... 
            // our algo treats them isloated nodes, but nxcore doesn't recognize them
        if(degrees[i]!=0)
           out<<'"'<<i<<'"'<<": "<<degrees[i]<<", "<<endl;
    out.seekp(-3, ios_base::end);
    out<<" }";
    out.close();
}

Graph::Graph(std::string input_file){
    if(readSerialized(input_file)) return;
    cout<<"Reading normal file... "<<endl;

    readFile(input_file);
    writeSerialized(input_file);
}

Graph::~Graph(){
    // cout<<"Deallocated... "<<endl;
    delete [] neighbors;
    delete [] neighbors_offset;
    delete [] degrees;
}