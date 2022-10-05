#include <vector>
#ifdef _WIN32
#include <hash_map>
#endif
#ifdef __linux__
#include <ext/hash_map>
#endif
#include "../MedusaRT/GraphGenerator.h"
#include "../MedusaRT/GraphConverter.h"
#include "../MultipleGPU/MultiGraphStorage.h"
#include "../MedusaRT/SystemLibCPU.h"
#include "../MultipleGPU/MultiPublicAPI.h"
#include "../MedusaRT/GraphGenerator.h"
#include "../MedusaRT/GraphReader.h"
#include "Configuration.h"
#include "../MultipleGPU/PartitionManager.h"
#include "../MultipleGPU/MultiUtilities.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include <fstream>


#include "../Tools/ReplicaNumberAnalysis.h"


#include<sys/time.h>


#define RUN_TIMES 1
#define CPU_COUNTER_PART


long get_cur_time()
{
	timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec*1000 + tv.tv_usec/1000);
}


int main(int argc, char **argv)
{

	long start1 = get_cur_time();

	if(argc != 5)
	{
		printf("Usage: Medusa GPU_Num Max_hop Gt_File_Name Partition_File_Name\n");
		exit(-1);
	}

	//global configuration
	InitConfig(argv[1], argv[2], EDGE_MESSAGE, true);
    MGLOBAL::combiner_datatype = CUDPP_FLOAT;
	MGLOBAL::combiner_operator = CUDPP_MIN;
	
	//print multiple hop statistics

	// float t;//timer value
	// unsigned int timer = 0;
	// cutCreateTimer(&timer);



	GTGraph gt_graph;
	//vector<GraphPartition> gp_array;
	GraphPartition *gp_array = new GraphPartition[MGLOBAL::num_gpu_to_use];
	gt_graph.get_parition(gp_array, argv[3], argv[4]);

	//set up data structures for all GPUs

	InitHostDS(gp_array);
	CheckOutgoingEdge(gp_array);
	printf("InitHostDS Done\n");



	/* <algorithm specific initialization>  */

    ifstream gt_file(argv[3]);
	
	int total_v_num = 0, total_e_num = 0;
	char first_ch;
	char line[1024];
	while(gt_file.get(first_ch))
	{
		if(first_ch == 'p')
		{
			string temp;
			gt_file>>temp>>total_v_num;
			gt_file.getline(line, 1024);//eat the line break
			break;
		}
		gt_file.getline(line, 1024);//eat the line break
	}

	int src_id, dst_id;
	float edge_weight;
	int *degree = (int *)malloc(sizeof(int)*(total_v_num+1));
	memset(degree, 0, sizeof(int)*(total_v_num+1));
	while(gt_file.get(first_ch))
	{
		if(first_ch == 'a')
		{
			gt_file>>src_id>>dst_id>>edge_weight;
			degree[src_id] ++;
		}
	}


    for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		medusaSetDevice(i);
		//initialize k-core values by its degree
		for(int vertex_index = 0; vertex_index < MGLOBAL::gpu_def[i].vertexArray.size; vertex_index++)
        {
            // printf("\n%d\n", degree[vertex_index+1]);
			MGLOBAL::gpu_def[i].vertexArray.level[vertex_index] = degree[vertex_index+1];
            MGLOBAL::gpu_def[i].vertexArray.updated[vertex_index] = true;
        }
	}
    free(degree);
	

	/*  </algorithm specific initialization>  */




	InitDeviceDS();

	//---------------------------------------------------------------------------------------------------------//


	long start2 = get_cur_time();



	//execute<PG loop>
	bool exe_temp, next_iter = true;
	int iter;
	// cutResetTimer(timer);

	while(next_iter)
	{
		//reset the global GPU side toNextIter variable
		MGLOBAL::toNextIter = false;
		ResetToNextIter();

		// printf("step = %d ----------------------------------------------------------\n",MGLOBAL::super_step);
		exe_temp = true;
		iter = 0;
		while(exe_temp)
		{
			// printf("\titer = %d ----------------------------------------------------------\n", iter++);
			//reset the global GPU side toExecute variable
			MGLOBAL::toExecute = false;
			ResetToExecute();

			Medusa_Exec();
			
		
			exe_temp = RetriveToExecute();
	
		}
		next_iter = RetriveToNextIter();

		(MGLOBAL::super_step) ++;
		if((MGLOBAL::super_step) >= 10000)
			break;
	}
	
	long end = get_cur_time();

	std::cout << "SuperStep : "<< (MGLOBAL::super_step) << std::endl;

    // record result
    // ofstream myfile;
    // myfile.open ("gpu_core");

    // VertexArray output;
    // MGLOBAL::gpu_def[0].d_vertexArray.Dump(output);
    
    // for(int i = 0; i < total_v_num; ++i)
    // {
    //     myfile << output.level[i] << "\n";
    // }
    // myfile.close();


	// t = cutGetTimerValue(timer);
	// printf("GPU KCORE %.3f ms\n",t); 
	printf("K-core GPU used %ld ms\n", (end - start2));
	printf("Total time used %ld ms\n", (end - start1));


	return 0;
}
