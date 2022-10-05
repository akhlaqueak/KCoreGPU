#ifndef VERTEXDATATYPE_H
#define VERTEXDATATYPE_H

//#include <cutil.h>

#include <helper_cuda.h>
#include "../Compatibility/Compatability.h"

#include <cuda_runtime.h>
#include "../MedusaRT/MessageArrayManager.h"
#include "../MedusaRT/GraphConverter.h"
#include "Configuration.h"


/**
* @dev under development, should be automatically generated
*/
struct VertexArray
{
	int *edge_count;
	MVT *level;
	int *msg_index;
	bool *updated;

	int *edge_index;

	bool *deleted;
	bool *to_be_deleted;
	int *degree;

	int size;
	VertexArray();
	/**
	* build the vertex array from a graph
	*/
	void build(GraphIR &graph);
	void resize(int num);
	void assign(int i, Vertex v);/* assign to element i of this array using a Vertex object */
};


/**
* @dev under development, should be automatically generated
*/
struct D_VertexArray
{
	int *d_edge_count;
	MVT *d_level;
	int *d_msg_index;
	bool *d_updated;

	int *d_edge_index;

	bool *d_deleted;
	bool *d_to_be_deleted;
	int *d_degree;

	int size;
	void Free();
	void Fill(VertexArray &varr);
	void Dump(VertexArray &varr);
};

#endif