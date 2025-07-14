#include "target_processing.cuh"
#include "../config/config.hpp"
#include <cmath>

namespace TargetProcessing {

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

    // Example: compute range, strength, etc. from peaksnaps
    double range = 0.0;
    double strength = 0.0;
    for (int r = 0; r < num_receivers; ++r) {
        int snap_idx = idx * num_receivers + r;
        strength += cuCabs(d_peaksnaps[snap_idx]);
    }
    // Dummy range calculation (replace with real logic)
    range = strength * 0.01;

    // Fill target struct
    d_targets[idx].x = range * cos(azimuth * M_PI / 180.0) * cos(elevation * M_PI / 180.0);
    d_targets[idx].y = range * sin(azimuth * M_PI / 180.0) * cos(elevation * M_PI / 180.0);
    d_targets[idx].z = range * sin(elevation * M_PI / 180.0);
    d_targets[idx].range = range;
    d_targets[idx].azimuth = azimuth;
    d_targets[idx].elevation = elevation;
    d_targets[idx].strength = strength;
    d_targets[idx].relativeSpeed = 0.0; // TODO: Doppler
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
