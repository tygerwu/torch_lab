#pragma once
#include "macro.cuh"
#include "device.cuh"
namespace LAB {

namespace CUDA {

template<typename Func>
static cudaError SelectBlockThreads(const Func& func,int dynamicSMemBytes,
                                    int* threadsPerBlock,int* blocksPerSM,bool print=false){
    
    int maxThreadsPerSM = GetMaxThreadsPerSM();
    int minThreadsPerBlock = maxThreadsPerSM / GetMaxBllocksPerSM();
    int maxThreadsPerBlock = GetMaxThreadsPerBlock();
    
    std::vector<int> threadsList;
    int threads = maxThreadsPerBlock;
    while(threads >= minThreadsPerBlock){
        threadsList.push_back(threads);
        threads /= 2;
    }
    
    float maxOccupancy = 0.01;
    for(auto threads : threadsList){
        int numBlocks = 0; 
        auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, thread, dynamicSMemBytes);
        if(status == cudaSuccess){
            float occupancy = ((float)numBlocks * thread) / maxThreadsPerSM;
            if(occupancy >= maxOccupancy){
                maxOccupancy = occupancy;
                if(thread > maxThreads){
                    *threadsPerBlock = threads;
                    *blocksPerSM = numBlocks;
                }
            }
            if(print){
                printf("Thread:%d,NumBlocks:%d,Occupancy:%.2f\n",threads,numBlocks,occupancy);
            }
        }
    }
}


}
}