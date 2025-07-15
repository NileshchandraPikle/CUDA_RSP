#include "rcs.cuh"
#include "../config/config.hpp"
#include "../cuda_utils/cuda_utils.hpp"
#include <cmath>      // For mathematical operations
#include <iostream>   // For debug output
#include <stdexcept>  // For std::runtime_error

namespace RCSEstimation {

__global__ void estimate_rcs_kernel(
    RadarData::Target* d_targets,
    int num_targets,
    double transmittedPower,
    double transmitterGain,
    double receiverGain,
    double wavelength)
{
    // Calculate thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check thread bounds
    if (idx >= num_targets) return;
    
    // Get target data
    double range = d_targets[idx].range;
    double receivedPower = d_targets[idx].strength;
    
    // Handle invalid range
    if (range <= 0.0 || isnan(range)) {
        d_targets[idx].rcs = 0.0;
        return;
    }
    
    // Calculate RCS using radar equation
    // RCS = (received_power * (4π)³ * range⁴) / (transmitted_power * tx_gain * rx_gain * λ²)
    double numerator = receivedPower * pow(4 * RadarConfig::PI, 3) * pow(range, 4);
    double denominator = transmittedPower * transmitterGain * receiverGain * pow(wavelength, 2);
    
    // Avoid division by zero
    if (denominator <= 1e-10) {
        d_targets[idx].rcs = 0.0;
    } else {
        d_targets[idx].rcs = numerator / denominator;
    }
    
    // Apply sanity check to RCS value (can be adjusted based on application needs)
    if (d_targets[idx].rcs < 0.0 || isinf(d_targets[idx].rcs)) {
        d_targets[idx].rcs = 0.0;
    }
}

void estimate_rcs_gpu(
    RadarData::TargetResults& targetResults,
    double transmittedPower,
    double transmitterGain,
    double receiverGain,
    double wavelength)
{
    try {
        // Parameter validation
        if (targetResults.d_targets == nullptr) {
            throw std::runtime_error("Invalid target data (null pointer)");
        }
        
        if (targetResults.num_targets <= 0) {
            std::cout << "No targets for RCS estimation" << std::endl;
            return;
        }
        
        if (transmittedPower <= 0.0 || wavelength <= 0.0) {
            throw std::runtime_error("Invalid radar parameters: transmittedPower and wavelength must be positive");
        }
        
        // Calculate kernel launch parameters
        const int threads_per_block = 256;
        int blocks = (targetResults.num_targets + threads_per_block - 1) / threads_per_block;
        
        std::cout << "RCS Estimation: Processing " << targetResults.num_targets 
                  << " targets with " << blocks << " blocks, " 
                  << threads_per_block << " threads per block" << std::endl;
        
        // Launch kernel
        estimate_rcs_kernel<<<blocks, threads_per_block>>>(
            targetResults.d_targets,
            targetResults.num_targets,
            transmittedPower,
            transmitterGain,
            receiverGain,
            wavelength);
        
        // Wait for kernel to complete
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Check for any kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        
    } catch (const std::exception& e) {
        std::cerr << "Error in RCS estimation: " << e.what() << std::endl;
        throw;
    }
}

}
