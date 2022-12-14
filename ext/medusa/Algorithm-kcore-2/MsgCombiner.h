#ifndef MSGCOMBINER_H
#define MSGCOMBINER_H

#include "Configuration.h"
#include "../MultipleGPU/MultiGraphStorage.h"


#ifdef __LINUX__
#include <sys/time.h>
#include <unistd.h>
#endif

#include <time.h>


//in millisecond
void printTimestamp()
{

#ifdef __LINUX__
	struct timeval  tv;
	struct timezone tz;


	struct tm      *tm;
	long long         start;

	gettimeofday(&tv, &tz);

	start = tv.tv_sec * 1000000 + tv.tv_usec;

	printf("%lld\n",start);
#endif


}


void *combine_reuse(void *did)
{
        int gpu_id = (long) did;
        if(cudaSetDevice(gpu_id) != cudaSuccess)
        {
                printf("combiner thread set device error (%d)\n", gpu_id);
                exit(-1);
        }

        while(true)
        {
                pthread_mutex_lock(&MGLOBAL::com.common_mutex);
                pthread_cond_wait(&MGLOBAL::com.common_condition, &MGLOBAL::com.common_mutex);
                pthread_mutex_unlock(&MGLOBAL::com.common_mutex);


                cudppSegmentedScan(MGLOBAL::gpu_def[gpu_id].scanplan, MGLOBAL::gpu_def[gpu_id].d_messageArrayBuf.d_val, MGLOBAL::gpu_def[gpu_id].d_messageArray.d_val, MGLOBAL::gpu_def[gpu_id].d_edgeArray.d_incoming_msg_flag, MGLOBAL::gpu_def[gpu_id].d_messageArray.size);
                MyCheckErrorMsg("after cudppSegmentedScan");


                cudaDeviceSynchronize();
                MyCheckErrorMsg("after sync cudppSegmentedScan");

                pthread_mutex_lock(&MGLOBAL::com.individual_mutex[gpu_id]);
                pthread_cond_signal(&MGLOBAL::com.individual_condition[gpu_id]);
                pthread_mutex_unlock(&MGLOBAL::com.individual_mutex[gpu_id]);

        }
        return 0;
}






void *combine(void *did)
{
	
	int gpu_id = (long) did;
	

	medusaSetDevice(gpu_id);

	
	// cudppSegmentedScan(MGLOBAL::gpu_def[gpu_id].scanplan, MGLOBAL::gpu_def[gpu_id].d_messageArrayBuf.d_val, MGLOBAL::gpu_def[gpu_id].d_messageArray.d_val, MGLOBAL::gpu_def[gpu_id].d_edgeArray.d_incoming_msg_flag, MGLOBAL::gpu_def[gpu_id].d_messageArray.size);
    
    // cudaMemcpy(MGLOBAL::gpu_def[gpu_id].d_messageArrayBuf.d_val, MGLOBAL::gpu_def[gpu_id].d_messageArray.d_val, MGLOBAL::gpu_def[gpu_id].d_messageArray.size, cudaMemcpyDeviceToDevice);


	MyCheckErrorMsg("after cudppSegmentedScan");

    cudaDeviceSynchronize();

	MyCheckErrorMsg("sync cudppSegmentedScan");

	return 0;
}

#endif
