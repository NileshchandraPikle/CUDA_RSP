#ifndef GHOST_REMOVAL_HPP
#define GHOST_REMOVAL_HPP

#include "../target_processing/target_processing.cuh"
#include "../data_types/datatypes.cuh"
#include "../cuda_utils/cuda_utils.hpp"
#include <cuda_runtime.h>

/**
 * @file ghost_removal.cuh
 * @brief Ghost target removal functionality for radar processing
 * 
 * This module provides functionality to filter out ghost targets that are
 * likely caused by reflections or other radar artifacts rather than
 * real physical objects.
 */

namespace GhostRemoval {
    /**
     * @brief Remove ghost targets based on relative speed difference from ego speed
     * 
     * This is the host implementation that processes targets on the CPU.
     * Ghost targets are identified by having a relative speed close to the ego speed.
     * 
     * @param targets Input target results containing all detected targets
     * @param egoSpeed Speed of the ego vehicle in m/s
     * @return RadarData::TargetResults Structure containing filtered targets
     */
    RadarData::TargetResults remove_ghost_targets(
        const RadarData::TargetResults& targets,
        double egoSpeed);

    /**
     * @brief GPU-parallel implementation of ghost target removal
     * 
     * This function uses CUDA to parallelize the ghost target removal process.
     * It first flags valid targets and then compacts them into a filtered list.
     * 
     * @param d_targets Input target array in device memory
     * @param num_targets Number of input targets
     * @param egoSpeed Speed of the ego vehicle in m/s
     * @param d_filtered Output array for filtered targets in device memory
     * @param d_num_filtered Pointer to store number of filtered targets (in device memory)
     * @param stream CUDA stream to use for operations (optional)
     * @throws std::runtime_error If CUDA operations fail
     */
    void remove_ghost_targets_gpu(
        const RadarData::Target* d_targets,
        int num_targets, 
        double egoSpeed,
        RadarData::Target* d_filtered, 
        int* d_num_filtered, 
        cudaStream_t stream = 0);
    
    /**
     * @brief Copy filtered targets from device to host memory
     * 
     * @param d_filtered Source array of filtered targets in device memory
     * @param num_filtered Number of filtered targets to copy
     * @param h_filtered Destination array in host memory
     * @throws std::runtime_error If CUDA operations fail
     */
    void copy_filtered_targets_to_host(
        const RadarData::Target* d_filtered, 
        int num_filtered, 
        RadarData::Target* h_filtered);
}

#endif // GHOST_REMOVAL_HPP
