#ifndef VERTEXDATATYPE_CU
#define VERTEXDATATYPE_CU

#include "VertexDataType.h"
#include "../MultipleGPU/MultiGraphStorage.h"
#include "../MedusaRT/Utilities.h"
#include "../MedusaRT/GraphConverter.h"

VertexArray::VertexArray()
{
	size = 0;
	level = NULL;
	msg_index = NULL;
	edge_count = NULL;
	updated = NULL;

	edge_index = NULL;

	deleted = NULL;
	to_be_deleted = NULL;
	degree = NULL;
}

void VertexArray::resize(int num)
{
	if(size)
	{
		free(level);
		free(msg_index);
		free(edge_count);
		free(updated);
		free(edge_index);
		free(deleted);
		free(to_be_deleted);
		free(degree);
	}
	size = num;
	CPUMalloc((void**)&level, sizeof(MVT)*size);
	CPUMalloc((void**)&msg_index, sizeof(int)*(size+1));
	CPUMalloc((void**)&edge_count, sizeof(int)*size);
	CPUMalloc((void**)&updated, sizeof(bool)*size);

	CPUMalloc((void**)&edge_index, sizeof(int)*(size+1));

	CPUMalloc((void**)&deleted, sizeof(bool)*size);
	CPUMalloc((void**)&to_be_deleted, sizeof(bool)*size);
	CPUMalloc((void**)&degree, sizeof(int)*size);
}

void VertexArray::assign(int i, Vertex v)
{
	level[i] = v.level;
	msg_index[i] = v.msg_index;
	edge_count[i] = v.edge_count;
	updated[i] = v.updated;
	//printf("edge_count[%d] = %d\n",i,edge_count[i]);
	edge_index[i] = v.edge_index;

	deleted[i] = false;
	to_be_deleted[i] = false;
	degree[i] = v.edge_count;

}

void VertexArray::build(GraphIR &graph)
{
	//construct vertex array
	resize(graph.vertexNum);
	for(int i = 0; i < graph.vertexNum; i ++)
	{
		assign(i, graph.vertexArray[i].vertex);
	}
	//construct msg_index
	//compute prefix sum
	
	msg_index[0] = 0;
	for(int i = 1; i <= graph.vertexNum; i ++)
		msg_index[i] = graph.vertexArray[i-1].incoming_edge_count + msg_index[i-1];		 

	edge_index[0] = 0;
	for(int i = 1; i <= graph.vertexNum; i ++)
	{
		edge_index[i] = graph.vertexArray[i-1].vertex.edge_count + edge_index[i-1];
	}
}


void D_VertexArray::Fill(VertexArray &varr)
{
	if(size != 0)
	{
		CUDA_SAFE_CALL(cudaFree(d_level));
		CUDA_SAFE_CALL(cudaFree(d_msg_index));
		CUDA_SAFE_CALL(cudaFree(d_edge_count));
		CUDA_SAFE_CALL(cudaFree(d_updated));
		CUDA_SAFE_CALL(cudaFree(d_edge_index));

		CUDA_SAFE_CALL(cudaFree(d_deleted));
		CUDA_SAFE_CALL(cudaFree(d_to_be_deleted));
		CUDA_SAFE_CALL(cudaFree(d_degree));

	}
	size = varr.size;
	GPUMalloc((void**)&d_level, sizeof(MVT)*size);
	GPUMalloc((void**)&d_msg_index, sizeof(int)*(size+1));
	GPUMalloc((void**)&d_edge_count, sizeof(int)*size);
	GPUMalloc((void**)&d_updated, sizeof(bool)*size);
	CUDA_SAFE_CALL(cudaMemcpy(d_level, varr.level, sizeof(MVT)*size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_msg_index, varr.msg_index, sizeof(int)*(size+1), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_edge_count, varr.edge_count, sizeof(int)*size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_updated, varr.updated, sizeof(bool)*size, cudaMemcpyHostToDevice));
    
	GPUMalloc((void**)&d_edge_index, sizeof(int)*(size+1));
	CUDA_SAFE_CALL(cudaMemcpy(d_edge_index, varr.edge_index, sizeof(int)*(size+1), cudaMemcpyHostToDevice));

	GPUMalloc((void**)&d_deleted, sizeof(bool)*(size));
	CUDA_SAFE_CALL(cudaMemcpy(d_deleted, varr.deleted, sizeof(bool)*size, cudaMemcpyHostToDevice));
	GPUMalloc((void**)&d_to_be_deleted, sizeof(bool)*(size));
	CUDA_SAFE_CALL(cudaMemcpy(d_to_be_deleted, varr.to_be_deleted, sizeof(bool)*size, cudaMemcpyHostToDevice));
	GPUMalloc((void**)&d_degree, sizeof(int)*(size));
	CUDA_SAFE_CALL(cudaMemcpy(d_degree, varr.degree, sizeof(int)*size, cudaMemcpyHostToDevice));

}


void D_VertexArray::Free()
{
	if(size != 0)
	{
		CUDA_SAFE_CALL(cudaFree(d_level));
		CUDA_SAFE_CALL(cudaFree(d_msg_index));
		CUDA_SAFE_CALL(cudaFree(d_edge_count));
		CUDA_SAFE_CALL(cudaFree(d_updated));
		CUDA_SAFE_CALL(cudaFree(d_edge_index));

		CUDA_SAFE_CALL(cudaFree(d_deleted));
		CUDA_SAFE_CALL(cudaFree(d_to_be_deleted));
		CUDA_SAFE_CALL(cudaFree(d_degree));
	}
}

void D_VertexArray::Dump(VertexArray &varr)
{

	if(size != varr.size)
	{
		if(varr.size)
		{
			free(varr.level);
			free(varr.msg_index);
			free(varr.edge_count);
			free(varr.updated);

			free(varr.deleted);
			free(varr.to_be_deleted);
			free(varr.degree);
		}
		CPUMalloc((void**)&varr.level, size*sizeof(MVT));
		CPUMalloc((void**)&varr.msg_index, size*sizeof(int));
		CPUMalloc((void**)&varr.edge_count, size*sizeof(int));
		CPUMalloc((void**)&varr.updated, size*sizeof(bool));

		CPUMalloc((void**)&varr.deleted, size*sizeof(bool));
		CPUMalloc((void**)&varr.to_be_deleted, size*sizeof(bool));
		CPUMalloc((void**)&varr.degree, size*sizeof(int));
	}
	CUDA_SAFE_CALL(cudaMemcpy(varr.level, d_level, sizeof(MVT)*size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(varr.msg_index, d_msg_index, sizeof(int)*size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(varr.edge_count, d_edge_count, sizeof(int)*size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(varr.updated, d_updated, sizeof(bool)*size, cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaMemcpy(varr.deleted, d_deleted, sizeof(bool)*size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(varr.to_be_deleted, d_to_be_deleted, sizeof(bool)*size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(varr.degree, d_degree, sizeof(int)*size, cudaMemcpyDeviceToHost));
}

#endif
