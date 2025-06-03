#include "peak_detection.cuh"
#include <cmath> // Include for std::abs
#include <tuple> // Include for std::make_tuple
#include <iostream> // Include for std::cout
#include "../config/config.hpp"
#include "../data_types/datatypes.cuh"
#include "../cuda_utils/cuda_utils.hpp"

namespace PeakDetection {
    // Function to perform 2D CFAR-like peak detection
    __global__ void cfar_peak_detection_Kernel(double* d_nci, double* d_foldedNci, double* d_noiseEstimation, double* d_thresholdingMap, RadarData::Peak* d_peakList,
        const cuDoubleComplex* d_data, int num_receivers, int num_chirps, int num_samples, double alpha, int max_num_peaks, int * d_peak_counter)
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
    void cfar_peak_detection(const RadarData::Frame& frame, RadarData::peakInfo& peakinfo) {
        // Initialize output structures
        int num_receivers = frame.num_receivers;
        int num_chirps = frame.num_chirps;
        int num_samples = frame.num_samples;
        peakinfo.num_peaks = 0;
        
        double alpha = RadarConfig::TRAINING_CELLS  * (std::pow(RadarConfig::FALSE_ALARM_RATE, -1.0 / RadarConfig::TRAINING_CELLS) - 1);
        CUDA_CHECK(cudaMemset(peakinfo.d_peak_counter, 0, sizeof(int)));
        dim3 block(3, 8, 8);
        dim3 grid((num_receivers + block.x - 1) / block.x, (num_chirps + block.y - 1) / block.y, (num_samples + block.z - 1) / block.z);


        cfar_peak_detection_Kernel<<<grid,block>>>(peakinfo.d_nci, peakinfo.d_foldedNci, peakinfo.d_noiseEstimation, peakinfo.d_thresholdingMap, peakinfo.d_peakList,
            frame.d_data, num_receivers, num_chirps, num_samples, alpha, peakinfo.max_num_peaks, peakinfo.d_peak_counter);
            cudaDeviceSynchronize();
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cfar_peak_detection_Kernel: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        CUDA_CHECK(cudaMemcpy(&peakinfo.num_peaks, peakinfo.d_peak_counter, sizeof(int), cudaMemcpyDeviceToHost));
        // This is required because in MIMO synthesis we need to allocate memory for the peak snaps
        // The memory allocation is done after the peak detection is done
    }// end of cfar_peak_detection
}// namespace PeakDetection
