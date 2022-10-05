
#include "EdgeDataType.h"
#include "../MedusaRT/GraphStorage.h"
#include "../MedusaRT/Utilities.h"


EdgeArray::EdgeArray()
{
	srcVertexID = NULL;
	dstVertexID = NULL;
	msgDstID = NULL;
	incoming_msg_flag = NULL;
	edgeOffset = NULL;
	level_count = 0;
	size = 0;
}

void EdgeArray::resize(int num)
{
	if(size != 0)
	{
		free(srcVertexID);
		free(dstVertexID);
		free(msgDstID);
		free(incoming_msg_flag);
	}
	size = num;
	CPUMalloc((void**)&srcVertexID,sizeof(int)*num);
	CPUMalloc((void**)&dstVertexID,sizeof(int)*num);
	CPUMalloc((void**)&msgDstID,sizeof(int)*num);
	CPUMalloc((void**)&incoming_msg_flag,sizeof(unsigned int)*num);
}

void EdgeArray::assign(int i, Edge e)
{
	srcVertexID[i] = e.srcVertexID;
	dstVertexID[i] = e.dstVertexID;
	msgDstID[i] = e.msgDstID;
}


void EdgeArray::buildAA(GraphIR &graph)
{
	resize(graph.totalEdgeCount);
	int placeIndex = 0;
	for(int i = 0; i < graph.vertexNum; i ++)
	{
		EdgeNode *tempEdgeNode = graph.vertexArray[i].firstEdge;
		while(tempEdgeNode != NULL)
		{
			//if(placeIndex == 130)
			//	printf("[%d]add edge (%d, %d)\n",placeIndex,tempEdgeNode->edge.srcVertexID, tempEdgeNode->edge.dstVertexID);
			assign(placeIndex, tempEdgeNode->edge);
			tempEdgeNode = tempEdgeNode->nextEdge;
			placeIndex ++;
		}
	}
	
	//compute incoming message flag for the combiner
	int *vertex_edge_count;
	CPUMalloc((void **)&vertex_edge_count, sizeof(int)*graph.vertexNum);
	for(int i = 0; i < graph.vertexNum; i ++)
	{
		vertex_edge_count[i] = graph.vertexArray[i].incoming_edge_count;
	}


	//compute prefix sum
	int last_edge_count = vertex_edge_count[0];
	vertex_edge_count[0] = 0;
	for(int i = 1; i < graph.vertexNum; i ++)
	{
		int temp_edge_count = vertex_edge_count[i];
		vertex_edge_count[i] = vertex_edge_count[i - 1] + last_edge_count;
		last_edge_count = temp_edge_count; 
	}


	memset(incoming_msg_flag, 0, sizeof(unsigned int)*size);
	for(int i = 0; i < graph.vertexNum; i ++)
	{
		if(vertex_edge_count[i] >= size)
		{
			//ignore the replica
			break;
		}
		incoming_msg_flag[vertex_edge_count[i]] = 1;
	}
	//compute reverse edge ID
	for(int i = 0; i < size; i ++)
	{
		//printf("%d %d\n",i,dstVertexID[i]);
		msgDstID[i] = vertex_edge_count[dstVertexID[i]] ++;
	}

	free(vertex_edge_count);

}


void D_EdgeArray::Fill(EdgeArray &ea)
{
	if(size != 0)
	{
		CUDA_SAFE_CALL(cudaFree(d_srcVertexID));
		CUDA_SAFE_CALL(cudaFree(d_dstVertexID));
		CUDA_SAFE_CALL(cudaFree(d_msgDstID));
		CUDA_SAFE_CALL(cudaFree(d_incoming_msg_flag));
	}
	size = ea.size;
	GPUMalloc((void**)&d_srcVertexID,sizeof(int)*size);
	GPUMalloc((void**)&d_dstVertexID,sizeof(int)*size);
	GPUMalloc((void**)&d_msgDstID,sizeof(int)*size);

	GPUMalloc((void**)&d_incoming_msg_flag,sizeof(unsigned int)*size);
	CUDA_SAFE_CALL(cudaMemcpy(d_srcVertexID, ea.srcVertexID, sizeof(int)*size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_dstVertexID, ea.dstVertexID, sizeof(int)*size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_msgDstID, ea.msgDstID, sizeof(int)*size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_incoming_msg_flag, ea.incoming_msg_flag, sizeof(unsigned int)*size, cudaMemcpyHostToDevice));
}
