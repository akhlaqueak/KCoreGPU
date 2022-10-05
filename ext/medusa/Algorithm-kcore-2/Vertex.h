#ifndef VERTEX_H
#define VERTEX_H 


#include "MessageDataType.h"
#include "Configuration.h"

struct Vertex
{
	int edge_count; // degree number
	MVT level; // message type, e.g., vertex's coreness

    bool updated;

    int msg_index; // starting index of vertex's message list
	int edge_index;

	bool deleted; 
	bool to_be_deleted;
	int degree; 
};


#endif