#include "rcs.cuh"
#include "../config/config.hpp"
#include <cmath> // For mathematical operations
#include <iostream> // For debug output

namespace RCSEstimation {

__global__ void estimate_rcs_kernel(
    RadarData::Target* d_targets,
    int num_targets,
    double transmittedPower,
    double transmitterGain,
    double receiverGain,
    double wavelength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_targets) return;
    double range = d_targets[idx].range;
    double receivedPower = d_targets[idx].strength;
    if (range <= 0.0) {
        d_targets[idx].rcs = 0.0;
        return;
    }
    d_targets[idx].rcs = (receivedPower * pow(4 * RadarConfig::PI, 3) * pow(range, 4)) /
        (transmittedPower * transmitterGain * receiverGain * pow(wavelength, 2));
}

void estimate_rcs_gpu(
    RadarData::TargetResults& targetResults,
    double transmittedPower,
    double transmitterGain,
    double receiverGain,
    double wavelength)
{
    int threads = 256;
    int blocks = (targetResults.num_targets + threads - 1) / threads;
    estimate_rcs_kernel<<<blocks, threads>>>(
        targetResults.d_targets,
        targetResults.num_targets,
        transmittedPower,
        transmitterGain,
        receiverGain,
        wavelength);
    cudaDeviceSynchronize();
}

}
