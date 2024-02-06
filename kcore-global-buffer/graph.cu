
#include "graph.h"


void Graph::allocateMemory(const Graph &g)
{
    chkerr(cudaMalloc(&(neighbors), g.neighbors_offset[g.V] * sizeof(unsigned int)));
    chkerr(cudaMemcpy(neighbors, g.neighbors, g.neighbors_offset[g.V] * sizeof(unsigned int), cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(neighbors_offset), (g.V + 1) * sizeof(unsigned int)));
    chkerr(cudaMemcpy(neighbors_offset, g.neighbors_offset, (g.V + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(degrees), (g.V) * sizeof(unsigned int)));
    chkerr(cudaMemcpy(degrees, g.degrees, (g.V) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    V = g.V;
}
bool Graph::readSerialized(string input_file, string prefix)
{
    input_file = input_file.substr(input_file.find_last_of("/") + 1);
    ifstream file;
    file.open(string(OUTPUT_LOC) + prefix + string("-") + input_file);
    if (file)
    {
        cout << "Reading serialized file... " << endl;
        file >> V;
        file >> E;
        degrees = new unsigned int[V];
        neighbors_offset = new unsigned int[V + 1];
        neighbors = new unsigned int[E];
        for (unsigned int i = 0; i < V; i++)
            file >> degrees[i];
        for (unsigned int i = 0; i < V + 1; i++)
            file >> neighbors_offset[i];
        for (unsigned int i = 0; i < E; i++)
            file >> neighbors[i];
        file.close();
        return true;
    }
    else
    {
        cout << prefix + string("-") + input_file << " : File couldn't open" << endl;
    }

    return false;
}

void Graph::print(){
    cout<<endl;
    for(unsigned int i=0;i<V;i++){
        cout<<i<<": ";
        for(unsigned int j=neighbors_offset[i];j<neighbors_offset[i+1]; j++){
            cout<<neighbors[j]<<" ";
        }
        cout<<endl;
    }
}

void Graph::writeSerialized(string input_file, string prefix)
{
    input_file = input_file.substr(input_file.find_last_of("/") + 1);

    ofstream file;
    file.open(string(OUTPUT_LOC) + prefix + string("-") + input_file);
    if (file)
    {
        file << V << endl;
        file << E << endl;
        for (unsigned int i = 0; i < V; i++)
            file << degrees[i] << endl;
        for (unsigned int i = 0; i < V + 1; i++)
            file << neighbors_offset[i] << ' ';
        for (unsigned int i = 0; i < E; i++)
            file << neighbors[i] << ' ';
        file.close();
    }
    else
    {
        cout << prefix + string("-") + input_file << " : File couldn't open" << endl;
    }
}
// Graph::Graph()
// {
//     // default constructor
// }
void Graph::readFile(string input_file)
{

    double load_start = omp_get_wtime();
    ifstream infile;
    infile.open(DS_LOC + input_file);
    if (!infile)
    {
        cout << "load graph file failed " << endl;
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
    // read number of nodes...
    string line;
    vector<pair<unsigned int, unsigned int>> lines;

    V = 0;
    while (std::getline(infile, line))
    {
        if (isalpha(line[0]))
            continue; // it's a comment
        std::istringstream iss(line);
        iss >> s >> t;
        if (s == t)
            continue; // remove self loops
        V = max(s, V);
        V = max(t, V);
        lines.push_back({s, t});
    }
    infile.close();

    V++; // vertices index starts from 0, so add 1 to number of vertices.
    vector<set<unsigned int>> ns(V);

    for (auto &p : lines)
    {
        ns[p.first].insert(p.second);
        ns[p.second].insert(p.first);
    }

    lines.clear();
    degrees = new unsigned int[V];
    for (unsigned int i = 0; i < V; i++)
    {
        degrees[i] = ns[i].size();
    }

    neighbors_offset = new unsigned int[V + 1];
    neighbors_offset[0] = 0;
    partial_sum(degrees, degrees + V, neighbors_offset + 1);

    E = neighbors_offset[V];
    neighbors = new unsigned int[E];

#pragma omp parallel for
    for (unsigned int i = 0; i < V; i++)
    {
        auto it = ns[i].begin();
        for (unsigned int j = neighbors_offset[i]; j < neighbors_offset[i + 1]; j++, it++)
            neighbors[j] = *it;
    }
    writeSerialized(input_file, "serialized");
}

void Graph::writeKCoreToDisk(std::string file)
{
    // writing kcore in json dictionary format
    std::ofstream out(OUTPUT_LOC + string("pkc-kcore-") + file);

    out << "{ ";

    for (unsigned long long int i = 0; i < V; ++i)
        // not writing zero degree nodes, because certain nodes in dataset are not present...
        // our algo treats them isloated nodes, but nxcore doesn't recognize them
        if (degrees[i] != 0)
            out << '"' << i << '"' << ": " << degrees[i] << ", " << endl;
    out.seekp(-3, ios_base::end);
    out << " }";
    out.close();
}

Graph::Graph(std::string input_file)
{
    if (readSerialized(input_file, "serialized"))
        return;
    cout << "Reading normal file... " << endl;

    auto start = chrono::steady_clock::now();
    readFile(input_file);
    auto end = chrono::steady_clock::now();

    cout << "File Loaded in: " << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << endl;
}


// Graph::~Graph()
// {
//     cout << "Deallocated... " << endl;
//     delete[] neighbors;
//     delete[] neighbors_offset;
//     delete[] degrees;
// }