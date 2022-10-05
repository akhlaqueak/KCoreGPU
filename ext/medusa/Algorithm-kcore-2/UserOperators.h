#ifndef USEROPERATORS_H
#define USEROPERATORS_H

#include "../MedusaRT/SystemLibGPU.h"
#include "Configuration.h"

//-------------------Medusa API Implementation------------------------------------------------------//

/**
* this function will run inside the BSP loop
* firstly user should define the functors (functor types: EDGE,EDGELIST,VERTEX,MESSAGE,MESSAGELIST)
* Then functos are passed to functor holders and user should indicate the functor type
* @param []	
* @return	
* @note	
*
*/

struct UpdateVertex
{
	__device__ void operator() (D_Vertex vertex, int super_step)
	{
		MVT msg = vertex.get_combined_msg();
		int K = super_step + 1;
		//printf("%d min dis = %f\n",vertex.index, msg_min);
		// if((int)msg != (int)vertex.get_level())
		// {
        //     // printf("msg=%d, get_level=%d", msg, vertex.get_level());
		// 	vertex.set_level(msg);
		// 	//if(vertex.index == 1)
		// 	//printf("update 1 to %f\n", msg_min);
		// 	vertex.set_updated(true);
		// 	Medusa_Continue();
		// }
		// else
		// 	vertex.set_updated(false);
		
		if (!vertex.get_deleted() && vertex.get_degree() > K)
		{
			vertex.decrement_degree(msg);
			if (vertex.get_degree() <= K)
			{
				Medusa_Continue();
			}
		}

		if (!vertex.get_deleted())
		{
			Medusa_NextIter();
		}
	}
};


//-------------------------------Testing Different Edge Representations----------------------------//
__global__ void SendMsgAA(int super_step, int total_thread_num)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int K = super_step + 1;
	for( ; tid < GRAPH_STORAGE_GPU::d_vertexArray.size; tid += total_thread_num)
	{
		//calculate message

		MVT msg;
		// if not deleted and degree <= K
		if (!GRAPH_STORAGE_GPU::d_vertexArray.d_deleted[tid] \
				&& GRAPH_STORAGE_GPU::d_vertexArray.d_degree[tid] <= K)
		{
			msg = 1;
			GRAPH_STORAGE_GPU::d_vertexArray.d_level[tid] = K;
			GRAPH_STORAGE_GPU::d_vertexArray.d_deleted[tid] = true;
		}
		else
		{
			msg = 0;
		}

		int start_index = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_index[tid];
		int end_index = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_index[tid + 1];
		//printf("start %d end %d\n",start_index, end_index);
		for(; start_index < end_index; start_index ++)
		{
			//send message
			int msg_dst_index = GRAPH_STORAGE_GPU::d_edgeArray.d_msgDstID[start_index];
			GRAPH_STORAGE_GPU::d_messageArray.d_val[msg_dst_index] = msg;
			//printf("[%d]=%f\n",msg_dst_index,msg);
		}
	}
}


void SM_EdgeList_Processor()
{

	int gridX = MGLOBAL::gpu_def[0].device_prop.multiProcessorCount*6; // to be changed!!
	SendMsgAA<<<gridX, 256>>>(MGLOBAL::super_step, gridX*256); // to be changed!!

	cudaDeviceSynchronize();
	MyCheckErrorMsg("after edge list");
}



//---------------------------------------------------------------------------------------------------------//
UpdateVertex uv;
Message init_msg;


// #define TIMEING_EACH_OPERATOR
void Medusa_Exec()
{
	init_msg.val = 0;
	InitMessageBuffer(init_msg);

	SM_EdgeList_Processor();
	
	//combiner
	MGLOBAL::com.combineAllDevice();

	FunctorHolder<VERTEX>::Run(uv);
}


#endif
