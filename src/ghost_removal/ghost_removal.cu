
#include "ghost_removal.cuh"
#include <cuda_runtime.h>
#include <cmath> // For std::abs

namespace GhostRemoval {

    RadarData::TargetResults remove_ghost_targets(
        const RadarData::TargetResults& targets,
        double egoSpeed) {
        RadarData::TargetResults filteredTargets(targets.num_targets); // Allocate with max possible size
        filteredTargets.num_targets = 0;
        constexpr double RELATIVE_SPEED_THRESHOLD = 5.0; // Example: 5 m/s
        for (int i = 0; i < targets.num_targets; i++) {
            const auto& target = targets.targets[i];  // Using targets instead of h_targets
            double relativeSpeedDifference = std::abs(target.relativeSpeed - egoSpeed);
            if (relativeSpeedDifference > RELATIVE_SPEED_THRESHOLD) {
                continue;
            }
            filteredTargets.targets[filteredTargets.num_targets++] = target;  // Using targets instead of h_targets
        }
        return filteredTargets;
    }

    // GPU-parallel implementation
    __global__ void ghost_removal_kernel(const RadarData::Target* targets, int num_targets, double egoSpeed, int* flags) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        constexpr double RELATIVE_SPEED_THRESHOLD = 5.0;
        if (idx < num_targets) {
            double diff = fabs(targets[idx].relativeSpeed - egoSpeed);
            flags[idx] = (diff <= RELATIVE_SPEED_THRESHOLD) ? 1 : 0;
        }
    }

    __global__ void compact_targets_kernel(const RadarData::Target* targets, const int* flags, int num_targets, RadarData::Target* filtered, int* num_filtered) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_targets && flags[idx]) {
            int out_idx = atomicAdd(num_filtered, 1);
            filtered[out_idx] = targets[idx];
        }
    }

    void remove_ghost_targets_gpu(const RadarData::Target* d_targets, int num_targets, double egoSpeed,
                                  RadarData::Target* d_filtered, int* d_num_filtered, cudaStream_t stream) {
        int blockSize = 256;
        int gridSize = (num_targets + blockSize - 1) / blockSize;
        int* d_flags = nullptr;
        cudaMalloc(&d_flags, num_targets * sizeof(int));
        cudaMemsetAsync(d_flags, 0, num_targets * sizeof(int), stream);
        cudaMemsetAsync(d_num_filtered, 0, sizeof(int), stream);
        ghost_removal_kernel<<<gridSize, blockSize, 0, stream>>>(d_targets, num_targets, egoSpeed, d_flags);
        compact_targets_kernel<<<gridSize, blockSize, 0, stream>>>(d_targets, d_flags, num_targets, d_filtered, d_num_filtered);
        cudaFree(d_flags);
    }

    void copy_filtered_targets_to_host(const RadarData::Target* d_filtered, int num_filtered, RadarData::Target* h_filtered) {
        cudaMemcpy(h_filtered, d_filtered, num_filtered * sizeof(RadarData::Target), cudaMemcpyDeviceToHost);
    }

} // namespace GhostRemoval
