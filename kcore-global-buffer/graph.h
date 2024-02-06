

#ifndef CUTS_GRAPH_H
#define CUTS_GRAPH_H
#include "./common.h"

class Graph{
public:
    unsigned int V;
    unsigned int E;
    unsigned int AVG_DEGREE = 0;
    unsigned int * neighbors;
    unsigned int * neighbors_offset;
    unsigned int * degrees;
    // Graph(const Graph& g){}
    Graph(std::string input_file);
    Graph(){};
    bool readSerialized(string input_file, string);
    void writeSerialized(string input_file, string);
    void readFile(string input_file);
    void writeKCoreToDisk(std::string file);
    void allocateMemory(const Graph& g);
    void print();
    // ~Graph();
};
#endif //CUTS_GRAPH_H
