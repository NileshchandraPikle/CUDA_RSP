#include "peak_detection.cuh"
#include <cmath> // Include for std::abs
#include <tuple> // Include for std::make_tuple
#include <iostream> // Include for std::cout
#include "../config/config.hpp"
#include "../data_types/datatypes.cuh"
#include "../cuda_utils/cuda_utils.hpp"

namespace PeakDetection {
    /**
     * CUDA kernel for 2D CFAR peak detection
     * 
     * This kernel performs Constant False Alarm Rate (CFAR) detection on radar data.
     * Each thread processes one cell and determines if it's a peak by comparing
     * its magnitude with the local noise level estimated from surrounding cells.
     * 
     * @param d_nci Output noise cell information
     * @param d_foldedNci Output folded noise cell information
     * @param d_noiseEstimation Output estimated noise levels
     * @param d_thresholdingMap Output threshold values
     * @param d_peakList Output array to store detected peaks
     * @param d_data Input radar data
     * @param num_receivers Number of receivers
     * @param num_chirps Number of chirps
     * @param num_samples Number of samples per chirp
     * @param alpha CFAR threshold factor
     * @param max_num_peaks Maximum number of peaks that can be stored
     * @param d_peak_counter Output counter for the number of peaks detected
     */
    __global__ void cfar_peak_detection_Kernel(
        double* d_nci, 
        double* d_foldedNci, 
        double* d_noiseEstimation,
        double* d_thresholdingMap, 
        RadarData::Peak* d_peakList,
        const cuDoubleComplex* d_data, 
        int num_receivers, 
        int num_chirps, 
        int num_samples, 
        double alpha, 
        int max_num_peaks, 
        int* d_peak_counter)
    {
            int r = blockIdx.x * blockDim.x + threadIdx.x;
            int c = blockIdx.y * blockDim.y + threadIdx.y;
            int s = blockIdx.z * blockDim.z + threadIdx.z;
            if (r >= num_receivers || c >= num_chirps || s >= num_samples) {
                return; // Out of bounds check
            }
            double magnitude = cuCabs(d_data[r * num_chirps * num_samples + c * num_samples + s]);
            // Calculate noise level using training cells in both Doppler and range dimensions
            double noise_level = 0.0;
            int training_count = 0;
            for (int tc = -RadarConfig::TRAINING_CELLS; tc <= RadarConfig::TRAINING_CELLS; tc++) {
                for (int ts = -RadarConfig::TRAINING_CELLS; ts <= RadarConfig::TRAINING_CELLS; ts++) {
                  if ((tc == 0 && ts == 0) ||
                    (abs(tc) <= RadarConfig::GUARD_CELLS && abs(ts) <= RadarConfig::GUARD_CELLS))
                    continue;

                 int dc = c + tc;
                 int rs = s + ts;

                    if (dc >= 0 && dc < num_chirps && rs >= 0 && rs < num_samples) {
                        int tidx = r * num_chirps * num_samples + dc * num_samples + rs;
                        noise_level += cuCabs(d_data[tidx]);
                        training_count++;
                    } // end of if condition
                } // end of ts loop
            } // end of tc loop

            double avg_noise = noise_level / training_count;
            double threshold = alpha * avg_noise;
            int idx = c * num_samples + s;
            d_noiseEstimation[idx] = avg_noise;
            d_thresholdingMap[idx] = threshold;
            d_nci[idx] = avg_noise;
            d_foldedNci[idx] = noise_level;

            if (magnitude > threshold) {
                unsigned int peak_id = atomicAdd(d_peak_counter, 1);
                d_peakList[peak_id] = {r, c, s, magnitude};
            }
        } // end of kernel function    
    /**
 * CFAR Peak Detection implementation for CUDA
 * 
 * This function detects peaks in radar data using the Constant False Alarm Rate (CFAR) algorithm.
 * The implementation uses a 2D CFAR approach examining both range and Doppler dimensions.
 * 
 * @param frame The radar frame data
 * @param peakinfo Structure to store peak detection results
 * @throws runtime_error if a CUDA error occurs
 */
void cfar_peak_detection(const RadarData::Frame& frame, RadarData::peakInfo& peakinfo) {
    try {
        // Initialize output structures
        int num_receivers = frame.num_receivers;
        int num_chirps = frame.num_chirps;
        int num_samples = frame.num_samples;
        peakinfo.num_peaks = 0;
        
        // Calculate CFAR threshold factor alpha based on false alarm rate
        double alpha = RadarConfig::TRAINING_CELLS * 
                     (std::pow(RadarConfig::FALSE_ALARM_RATE, -1.0 / RadarConfig::TRAINING_CELLS) - 1);
        
        // Reset peak counter
        CUDA_CHECK(cudaMemset(peakinfo.d_peak_counter, 0, sizeof(int)));
        
        // Configure kernel execution parameters
        dim3 block(3, 8, 8); // Thread block dimensions
        dim3 grid(
            (num_receivers + block.x - 1) / block.x,
            (num_chirps + block.y - 1) / block.y, 
            (num_samples + block.z - 1) / block.z
        );
        
        // Execute peak detection kernel
        cfar_peak_detection_Kernel<<<grid, block>>>(
            peakinfo.d_nci, 
            peakinfo.d_foldedNci, 
            peakinfo.d_noiseEstimation,
            peakinfo.d_thresholdingMap, 
            peakinfo.d_peakList,
            frame.d_data, 
            num_receivers, 
            num_chirps, 
            num_samples, 
            alpha, 
            peakinfo.max_num_peaks, 
            peakinfo.d_peak_counter
        );
        
        // Wait for kernel execution to complete
        cudaDeviceSynchronize();
        
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error in cfar_peak_detection_Kernel: ") + 
                                     cudaGetErrorString(err));
        }
        
        // Copy peak count from device to host
        CUDA_CHECK(cudaMemcpy(&peakinfo.num_peaks, peakinfo.d_peak_counter, 
                             sizeof(int), cudaMemcpyDeviceToHost));
                             
        // Ensure peak count doesn't exceed maximum
        if (peakinfo.num_peaks > peakinfo.max_num_peaks) {
            std::cerr << "Warning: Peak count exceeds maximum (" 
                      << peakinfo.num_peaks << " > " << peakinfo.max_num_peaks 
                      << "). Truncating to maximum." << std::endl;
            peakinfo.num_peaks = peakinfo.max_num_peaks;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in cfar_peak_detection: " << e.what() << std::endl;
        throw;
    }
}// end of cfar_peak_detection
}// namespace PeakDetection
