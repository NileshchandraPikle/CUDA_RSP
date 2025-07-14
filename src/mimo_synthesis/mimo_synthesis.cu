#include <iostream>
#include <vector>
#include <complex>
#include "../config/config.hpp"
#include "mimo_synthesis.cuh"

namespace MIMOSynthesis {

    __global__ void synthesize_peaks_kernel(const cuDoubleComplex* d_data, RadarData::Peak* d_peakList,cuDoubleComplex* d_peaksnaps, int num_peaks, int num_receivers, int num_chirps, int num_samples, int max_num_peaks) {
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
        
        for (int r = 0; r < num_receivers; ++r) {
            int i = r * num_chirps * num_samples + peak.chirp * num_samples + peak.sample;
            d_peaksnaps[idx * num_receivers + r] = d_data[i];
        }
    }
    void synthesize_peaks(const RadarData::Frame& frame, RadarData::peakInfo& peakinfo) {
        // Clear the output PeakSnaps
        peakinfo.initializePeakSnaps();
        int blocks = (peakinfo.num_peaks + 255) / 256; // Calculate number of blocks
        int threads = 256; // Number of threads per block
        std::cout << "Number of blocks: " << blocks << ", Threads per block: " << threads << std::endl;
        synthesize_peaks_kernel<<<blocks,threads>>>(frame.d_data, peakinfo.d_peakList,peakinfo.d_peaksnaps, peakinfo.num_peaks, frame.num_receivers, frame.num_chirps, frame.num_samples, peakinfo.max_num_peaks);
        cudaDeviceSynchronize();
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in synthesize_peaks_kernel: " << cudaGetErrorString(err) << std::endl;
            return;
        }
    }
}
