#pragma once
#include <cuda_runtime.h>
#include "utils.cuh"


static void PrintDeviceInfo(){
    int32_t device{};
    CUDA_ERROR_CHECK(cudaGetDevice(&device)); 

    cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "===Device Info===" << std::endl;
    std::cout << "Selected Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000000.F << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << std::endl;
    std::cout << "Peak Memory Bandwidth: " << 2.F * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1000000.F << std::endl;

    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads per MultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max Blocks per MultiProcessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Max Shared Memory per Block: " << prop.sharedMemPerBlock << std::endl;
    std::cout << "Max Shared Memory per MultiProcessor: " << prop.sharedMemPerMultiprocessor << std::endl;
    std::cout << "Max Registers per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Max Registers per MultiProcessor: " << prop.regsPerMultiprocessor << std::endl;

    std::cout << "Max Thread Dimensions: " << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max Grid Dimensions: " << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << std::endl;

}