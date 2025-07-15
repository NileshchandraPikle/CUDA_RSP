
#include "ghost_removal.cuh"
#include <cuda_runtime.h>
#include <cmath>       // For std::abs
#include <iostream>    // For std::cout
#include <stdexcept>   // For std::runtime_error

namespace GhostRemoval {

    // Configuration constant
    constexpr double RELATIVE_SPEED_THRESHOLD = 5.0; // 5 m/s threshold to identify ghost targets

    RadarData::TargetResults remove_ghost_targets(
        const RadarData::TargetResults& targets,
        double egoSpeed) {
        
        // Validate input
        if (targets.num_targets <= 0) {
            std::cout << "No targets to filter in ghost removal" << std::endl;
            return RadarData::TargetResults(0);
        }
        
        // Allocate with max possible size - using only host memory to prevent double-free issues
        RadarData::TargetResults filteredTargets(targets.num_targets);
        filteredTargets.free_device(); // Free device memory to avoid leaks
        filteredTargets.num_targets = 0;
        
        std::cout << "Ghost removal: Processing " << targets.num_targets << " targets" << std::endl;
        int removed = 0;
        
        // Process each target
        for (int i = 0; i < targets.num_targets; i++) {
            const auto& target = targets.targets[i];
            double relativeSpeedDifference = std::abs(target.relativeSpeed - egoSpeed);
            
            // Filter out ghost targets (those with relative speed close to ego speed)
            if (relativeSpeedDifference > RELATIVE_SPEED_THRESHOLD) {
                removed++;
                continue;
            }
            
            // Keep valid targets
            filteredTargets.targets[filteredTargets.num_targets++] = target;
        }
        
        std::cout << "Ghost removal: Removed " << removed << " targets, kept " 
                  << filteredTargets.num_targets << " targets" << std::endl;
                  
        return filteredTargets;
    }

    /**
     * @brief CUDA kernel to flag valid (non-ghost) targets
     * 
     * This kernel marks targets as valid (1) or ghost (0) based on the
     * difference between their relative speed and the ego speed.
     * 
     * @param targets Input array of targets
     * @param num_targets Number of targets
     * @param egoSpeed Ego vehicle speed
     * @param flags Output array of flags (1 = keep, 0 = remove)
     */
    __global__ void ghost_removal_kernel(
        const RadarData::Target* targets, 
        int num_targets, 
        double egoSpeed, 
        int* flags) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_targets) return;
        
        // Calculate speed difference and determine if target is valid
        double diff = fabs(targets[idx].relativeSpeed - egoSpeed);
        flags[idx] = (diff > RELATIVE_SPEED_THRESHOLD) ? 0 : 1;
    }

    /**
     * @brief CUDA kernel to compact valid targets into a contiguous array
     * 
     * This kernel uses atomic operations to create a compact list of
     * valid targets in the output array.
     * 
     * @param targets Input array of all targets
     * @param flags Array of flags indicating valid targets (1 = keep)
     * @param num_targets Number of input targets
     * @param filtered Output array for filtered targets
     * @param num_filtered Pointer to store number of filtered targets
     */
    __global__ void compact_targets_kernel(
        const RadarData::Target* targets, 
        const int* flags, 
        int num_targets, 
        RadarData::Target* filtered, 
        int* num_filtered) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_targets) return;
        
        // If this target is valid, add it to the filtered list
        if (flags[idx]) {
            int out_idx = atomicAdd(num_filtered, 1);
            filtered[out_idx] = targets[idx];
        }
    }

    void remove_ghost_targets_gpu(
        const RadarData::Target* d_targets, 
        int num_targets, 
        double egoSpeed,
        RadarData::Target* d_filtered, 
        int* d_num_filtered, 
        cudaStream_t stream) {
        
        try {
            // Parameter validation
            if (d_targets == nullptr || d_filtered == nullptr || d_num_filtered == nullptr) {
                throw std::runtime_error("Invalid pointers in ghost target removal");
            }
            
            if (num_targets <= 0) {
                std::cout << "No targets to filter in GPU ghost removal" << std::endl;
                CUDA_CHECK(cudaMemsetAsync(d_num_filtered, 0, sizeof(int), stream));
                return;
            }
            
            // Calculate kernel launch parameters
            const int threads_per_block = 256;
            int blocks = (num_targets + threads_per_block - 1) / threads_per_block;
            
            // Allocate and initialize device memory for flags
            int* d_flags = nullptr;
            CUDA_CHECK(cudaMalloc(&d_flags, num_targets * sizeof(int)));
            CUDA_CHECK(cudaMemsetAsync(d_flags, 0, num_targets * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(d_num_filtered, 0, sizeof(int), stream));
            
            std::cout << "GPU Ghost Removal: Processing " << num_targets 
                      << " targets with " << blocks << " blocks, " 
                      << threads_per_block << " threads per block" << std::endl;
            
            // Launch kernels
            ghost_removal_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_targets, num_targets, egoSpeed, d_flags);
                
            compact_targets_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_targets, d_flags, num_targets, d_filtered, d_num_filtered);
            
            // Check for kernel errors
            CUDA_CHECK(cudaGetLastError());
            
            // Free temporary device memory
            CUDA_CHECK(cudaFree(d_flags));
            
        } catch (const std::exception& e) {
            std::cerr << "Error in ghost target removal: " << e.what() << std::endl;
            throw;
        }
    }

    void copy_filtered_targets_to_host(
        const RadarData::Target* d_filtered, 
        int num_filtered, 
        RadarData::Target* h_filtered) {
        
        try {
            // Parameter validation
            if (d_filtered == nullptr || h_filtered == nullptr) {
                throw std::runtime_error("Invalid pointers in copy_filtered_targets_to_host");
            }
            
            if (num_filtered <= 0) {
                std::cout << "No filtered targets to copy to host" << std::endl;
                return;
            }
            
            // Copy data from device to host
            CUDA_CHECK(cudaMemcpy(h_filtered, d_filtered, 
                                  num_filtered * sizeof(RadarData::Target), 
                                  cudaMemcpyDeviceToHost));
                                  
        } catch (const std::exception& e) {
            std::cerr << "Error copying filtered targets to host: " << e.what() << std::endl;
            throw;
        }
    }

} // namespace GhostRemoval
