#ifndef GHOST_REMOVAL_HPP
#define GHOST_REMOVAL_HPP

#include "../target_processing/target_processing.cuh"
#include "../data_types/datatypes.cuh"
#include <cuda_runtime.h>

namespace GhostRemoval {
    // Function to remove ghost targets
    RadarData::TargetResults remove_ghost_targets(
        const RadarData::TargetResults& targets,
        double egoSpeed);

    // GPU-parallel ghost removal
    void remove_ghost_targets_gpu(const RadarData::Target* d_targets, int num_targets, double egoSpeed,
                                  RadarData::Target* d_filtered, int* d_num_filtered, cudaStream_t stream = 0);
    void copy_filtered_targets_to_host(const RadarData::Target* d_filtered, int num_filtered, RadarData::Target* h_filtered);
}

#endif // GHOST_REMOVAL_HPP
