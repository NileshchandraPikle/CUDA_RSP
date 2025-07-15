
#ifndef TARGET_PROCESSING_CUH
#define TARGET_PROCESSING_CUH

#include <stdexcept>
#include "../data_types/datatypes.cuh"

/**
 * @namespace TargetProcessing
 * @brief Contains functions and kernels for radar target detection
 */
namespace TargetProcessing {
    /**
     * @brief CUDA kernel for target detection
     * 
     * Detects targets based on peak snapshots and DoA information.
     * Calculates target position, range, strength, and relative speed.
     * 
     * @param d_peaksnaps Input peak snapshots on device
     * @param d_angles Input DoA angles on device
     * @param num_peaks Number of peaks to process
     * @param num_receivers Number of receivers
     * @param d_targets Output target data on device
     */
    __global__ void detect_targets_kernel(
        const cuDoubleComplex* d_peaksnaps,
        const RadarData::DoAangles* d_angles,
        int num_peaks,
        int num_receivers,
        RadarData::Target* d_targets
    );

    /**
     * @brief Host wrapper for GPU-based target detection
     * 
     * Coordinates the execution of the target detection kernel on GPU.
     * Handles error checking and parameter validation.
     * 
     * @param d_peaksnaps Input peak snapshots on device
     * @param d_angles Input DoA angles on device
     * @param num_peaks Number of peaks to process
     * @param num_receivers Number of receivers
     * @param targetResults Output structure for detected targets
     * @throws std::runtime_error if a CUDA error occurs
     */
    void detect_targets_gpu(
        cuDoubleComplex* d_peaksnaps,
        RadarData::DoAangles* d_angles,
        int num_peaks,
        int num_receivers,
        RadarData::TargetResults& targetResults
    );
}

#endif // TARGET_PROCESSING_CUH
