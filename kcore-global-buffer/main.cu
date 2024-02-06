#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include "common.h"
#include "graph.h"
#include "KCore.h"



int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cout << "Please provide data file" << endl;
        exit(-1);
    }
    std::string data_file = argv[1];


    cout << "Loading Started" << endl;
    Graph g(data_file);
    cout << "Loading Done" << endl;

    Degeneracy deg(g);
    deg.degenerate(); // performs KCore Decomposition, and rec is sorted in degeneracy order
// #ifdef DEGREESORT
//     deg.degreeSort();        // sorts the vertices based on degrees, within the degeneracy order
// #endif
//     Graph gRec = deg.recode();

    cout << "BKEX: " << data_file <<endl;
    return 0;
}

