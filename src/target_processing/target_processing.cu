#include "target_processing.cuh"
#include "../config/config.hpp"
#include <cmath>

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

void detect_targets_gpu(
    cuDoubleComplex* d_peaksnaps,
    RadarData::DoAangles* d_angles,
    int num_peaks,
    int num_receivers,
    RadarData::TargetResults& targetResults
) {
    int threads = 256;
    int blocks = (num_peaks + threads - 1) / threads;
    detect_targets_kernel<<<blocks, threads>>>(
        d_peaksnaps,
        d_angles,
        num_peaks,
        num_receivers,
        targetResults.d_targets
    );
    cudaDeviceSynchronize();
}

} // namespace TargetProcessing
