#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include "../config/config.hpp"
#include "../cuda_utils/cuda_utils.hpp"
#include "mimo_synthesis.cuh"

namespace MIMOSynthesis {

    __global__ void synthesize_peaks_kernel(
            const cuDoubleComplex* d_data,
            RadarData::Peak* d_peakList,
            cuDoubleComplex* d_peaksnaps,
            int num_peaks,
            int num_receivers,
            int num_chirps,
            int num_samples,
            int max_num_peaks) {
            
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_peaks) return;
        
        // Extract peak information
        RadarData::Peak peak = d_peakList[idx];
        
        // Validate indices
        if (peak.receiver < 0 || peak.receiver >= num_receivers ||
            peak.chirp < 0 || peak.chirp >= num_chirps ||
            peak.sample < 0 || peak.sample >= num_samples) {
            return; // Invalid peak indices
        }
        
        // Extract peak data from all receivers
        for (int r = 0; r < num_receivers; ++r) {
            // Calculate linear index in the data array
            int i = r * num_chirps * num_samples + peak.chirp * num_samples + peak.sample;
            
            // Store the complex value from this receiver in the peak snapshot
            d_peaksnaps[idx * num_receivers + r] = d_data[i];
        }
    }

    void synthesize_peaks(const RadarData::Frame& frame, RadarData::peakInfo& peakinfo) {
        try {
            // Parameter validation
            if (frame.d_data == nullptr) {
                throw std::runtime_error("Invalid radar frame data (null pointer)");
            }
            
            if (peakinfo.d_peakList == nullptr) {
                throw std::runtime_error("Invalid peak list data (null pointer)");
            }
            
            if (peakinfo.num_peaks <= 0) {
                std::cout << "No peaks to process in MIMO synthesis" << std::endl;
                return;
            }
            
            // Clear the output PeakSnaps
            peakinfo.initializePeakSnaps();
            
            // Calculate kernel launch parameters
            const int threads_per_block = 256;
            int blocks = (peakinfo.num_peaks + threads_per_block - 1) / threads_per_block;
            
            std::cout << "MIMO Synthesis: Processing " << peakinfo.num_peaks 
                      << " peaks with " << blocks << " blocks, " 
                      << threads_per_block << " threads per block" << std::endl;
            
            // Launch kernel
            synthesize_peaks_kernel<<<blocks, threads_per_block>>>(
                frame.d_data,
                peakinfo.d_peakList,
                peakinfo.d_peaksnaps,
                peakinfo.num_peaks,
                frame.num_receivers,
                frame.num_chirps,
                frame.num_samples,
                peakinfo.max_num_peaks
            );
            
            // Wait for kernel to complete
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Check for any kernel launch errors
            CUDA_CHECK(cudaGetLastError());
            
        } catch (const std::exception& e) {
            std::cerr << "Error in MIMO synthesis: " << e.what() << std::endl;
            throw;
        }
    }
}
