#include "target_processing.cuh"
#include "../config/config.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace TargetProcessing {
/*
 * This file contains the CUDA implementation of target detection.
 * It has been updated to match the sequential implementation logic:
 * 1. Proper range calculation using time delay estimation
 * 2. Correct coordinate calculation formulas
 * 3. Implementation of Doppler shift calculation for relative speed
 */

// Corrected kernel to match sequential implementation
__global__ void detect_targets_kernel(
    const cuDoubleComplex* d_peaksnaps,
    const RadarData::DoAangles* d_angles,
    int num_peaks,
    int num_receivers,
    RadarData::Target* d_targets
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_peaks) return;

    // Example: extract azimuth/elevation from DoAangles
    double azimuth = d_angles[idx].azimuth;
    double elevation = d_angles[idx].elevation;

    // Calculate signal strength first
    double strength = 0.0;
    for (int r = 0; r < num_receivers; ++r) {
        int snap_idx = idx * num_receivers + r;
        strength += cuCabs(d_peaksnaps[snap_idx]);
    }
    
    // Calculate range using time delay from first sample's magnitude
    // Using same approach as sequential version's calculate_time_delay function
    double c = 3e8; // Speed of light in m/s
    double timeDelay = cuCabs(d_peaksnaps[idx * num_receivers]) * 1e-9;
    double range = (c * timeDelay) / 2.0;

    // Convert azimuth and elevation to radians (as in sequential version)
    double azimuthRad = azimuth * M_PI / 180.0;
    double elevationRad = elevation * M_PI / 180.0;
    
    // Fill target struct with corrected coordinate calculation (matching sequential version)
    d_targets[idx].x = range * cos(elevationRad) * cos(azimuthRad);
    d_targets[idx].y = range * cos(elevationRad) * sin(azimuthRad);
    d_targets[idx].z = range * sin(elevationRad);
    d_targets[idx].range = range;
    d_targets[idx].azimuth = azimuth;
    d_targets[idx].elevation = elevation;
    d_targets[idx].strength = strength;
    
    // Calculate Doppler shift - implementation equivalent to sequential version
    double dopplerShift = 0.0;
    int validPhases = 0;
    for (int r = 1; r < num_receivers; ++r) {
        int cur_idx = idx * num_receivers + r;
        int prev_idx = idx * num_receivers + r - 1;
        double phase_cur = atan2(d_peaksnaps[cur_idx].y, d_peaksnaps[cur_idx].x);
        double phase_prev = atan2(d_peaksnaps[prev_idx].y, d_peaksnaps[prev_idx].x);
        dopplerShift += phase_cur - phase_prev;
        validPhases++;
    }
    
    if (validPhases > 0) {
        dopplerShift /= validPhases; // Average phase difference
    }
    
    // Calculate relative speed using Doppler shift
    double wavelength = RadarConfig::WAVELENGTH;
    d_targets[idx].relativeSpeed = (dopplerShift * wavelength) / 2.0;
}

/**
 * GPU implementation of target detection
 * 
 * This function detects targets based on peak snapshots and direction of arrival information.
 * It uses our corrected implementation that matches the sequential version.
 * 
 * @param d_peaksnaps Input peak snapshot data on device
 * @param d_angles Input direction of arrival angles on device
 * @param num_peaks Number of peaks to process
 * @param num_receivers Number of receivers
 * @param targetResults Output structure for detected targets
 * @throws runtime_error if a CUDA error occurs
 */
void detect_targets_gpu(
    cuDoubleComplex* d_peaksnaps,
    RadarData::DoAangles* d_angles,
    int num_peaks,
    int num_receivers,
    RadarData::TargetResults& targetResults
) {
    try {
        // Validate input parameters
        if (num_peaks <= 0) {
            std::cerr << "Warning: No peaks to process in target detection" << std::endl;
            targetResults.num_targets = 0;
            return;
        }
        
        if (!d_peaksnaps || !d_angles) {
            throw std::runtime_error("Invalid device pointers provided to target detection");
        }
        
        // Set kernel execution parameters
        int threads = 256;
        int blocks = (num_peaks + threads - 1) / threads;
        
        // Launch target detection kernel
        detect_targets_kernel<<<blocks, threads>>>(
            d_peaksnaps,
            d_angles,
            num_peaks,
            num_receivers,
            targetResults.d_targets
        );
        
        // Wait for kernel execution to complete
        cudaDeviceSynchronize();
        
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error in detect_targets_kernel: ") + 
                                     cudaGetErrorString(err));
        }
        
        // Set number of detected targets
        targetResults.num_targets = num_peaks;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in detect_targets_gpu: " << e.what() << std::endl;
        throw;
    }
}

} // namespace TargetProcessing
