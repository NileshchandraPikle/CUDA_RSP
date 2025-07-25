#include "batch_peak_detection.cuh"
#include <cmath>
#include <iostream>
#include "../config/config.hpp"
#include "../data_types/datatypes.cuh"
#include "../cuda_utils/cuda_utils.hpp"

namespace BatchPeakDetection {

// Initialize persistent device pointer arrays
const cuDoubleComplex** d_persistent_data_array = nullptr;
double** d_persistent_nci_array = nullptr;
double** d_persistent_foldedNci_array = nullptr;
double** d_persistent_noiseEstimation_array = nullptr;
double** d_persistent_thresholdingMap_array = nullptr;
RadarData::Peak** d_persistent_peakList_array = nullptr;
int** d_persistent_peak_counter_array = nullptr;
bool persistent_arrays_initialized = false;
int persistent_arrays_size = 0;

/**
 * CUDA kernel for batch CFAR peak detection
 * 
 * This kernel processes multiple frames in parallel, with each frame being processed
 * by a different thread block group. Within each frame, the CFAR algorithm is applied
 * to detect peaks in the radar data.
 */
__global__ void batchCfarPeakDetectionKernel(
    double** d_nci_array,
    double** d_foldedNci_array,
    double** d_noiseEstimation_array,
    double** d_thresholdingMap_array,
    RadarData::Peak** d_peakList_array,
    const cuDoubleComplex** d_data_array,
    int** d_peak_counter_array,
    int num_receivers,
    int num_chirps,
    int num_samples,
    double alpha,
    int max_num_peaks,
    int numFrames
) {
    // Calculate the frame index based on block index
    int frameIdx = blockIdx.x / ((num_receivers + blockDim.x - 1) / blockDim.x);
    
    // If frameIdx is out of bounds, return
    if (frameIdx >= numFrames) return;
    
    // Adjust blockIdx.x to be relative to the current frame
    int adjustedBlockIdx = blockIdx.x % ((num_receivers + blockDim.x - 1) / blockDim.x);
    
    // Calculate the cell position within the frame
    int r = adjustedBlockIdx * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Out of bounds check
    if (r >= num_receivers || c >= num_chirps || s >= num_samples) {
        return;
    }
    
    // Get the data pointer for the current frame
    const cuDoubleComplex* d_data = d_data_array[frameIdx];
    double* d_nci = d_nci_array[frameIdx];
    double* d_foldedNci = d_foldedNci_array[frameIdx];
    double* d_noiseEstimation = d_noiseEstimation_array[frameIdx];
    double* d_thresholdingMap = d_thresholdingMap_array[frameIdx];
    RadarData::Peak* d_peakList = d_peakList_array[frameIdx];
    int* d_peak_counter = d_peak_counter_array[frameIdx];
    
    // Calculate magnitude of the current cell
    double magnitude = cuCabs(d_data[r * num_chirps * num_samples + c * num_samples + s]);
    
    // Calculate noise level using training cells in both Doppler and range dimensions
    double noise_level = 0.0;
    int training_count = 0;
    
    for (int tc = -RadarConfig::TRAINING_CELLS; tc <= RadarConfig::TRAINING_CELLS; tc++) {
        for (int ts = -RadarConfig::TRAINING_CELLS; ts <= RadarConfig::TRAINING_CELLS; ts++) {
            // Skip guard cells and the cell under test
            if ((tc == 0 && ts == 0) ||
                (abs(tc) <= RadarConfig::GUARD_CELLS && abs(ts) <= RadarConfig::GUARD_CELLS))
                continue;

            int dc = c + tc;
            int rs = s + ts;

            // Check if the training cell is within bounds
            if (dc >= 0 && dc < num_chirps && rs >= 0 && rs < num_samples) {
                int tidx = r * num_chirps * num_samples + dc * num_samples + rs;
                noise_level += cuCabs(d_data[tidx]);
                training_count++;
            }
        }
    }

    // Calculate average noise and threshold
    double avg_noise = noise_level / training_count;
    double threshold = alpha * avg_noise;
    
    // Store results in output arrays
    int idx = c * num_samples + s;
    d_noiseEstimation[idx] = avg_noise;
    d_thresholdingMap[idx] = threshold;
    d_nci[idx] = avg_noise;
    d_foldedNci[idx] = noise_level;

    // If the cell is a peak, add it to the peak list
    if (magnitude > threshold) {
        unsigned int peak_id = atomicAdd(d_peak_counter, 1);
        if (peak_id < max_num_peaks) {
            d_peakList[peak_id] = {r, c, s, magnitude};
        }
    }
}

/**
 * Initialize persistent device arrays for batch processing
 */
bool initializePersistentArrays(
    const std::vector<RadarData::Frame>& frames,
    const std::vector<RadarData::peakInfo>& peakInfos,
    int numFrames
) {
    try {
        // Clean up any existing arrays first
        if (persistent_arrays_initialized) {
            cleanupPersistentArrays();
        }
        
        std::cout << "Initializing persistent device arrays for batch peak detection..." << std::endl;
        
        // Create host arrays to hold pointers
        cuDoubleComplex** h_data_array = new cuDoubleComplex*[numFrames];
        double** h_nci_array = new double*[numFrames];
        double** h_foldedNci_array = new double*[numFrames];
        double** h_noiseEstimation_array = new double*[numFrames];
        double** h_thresholdingMap_array = new double*[numFrames];
        RadarData::Peak** h_peakList_array = new RadarData::Peak*[numFrames];
        int** h_peak_counter_array = new int*[numFrames];
        
        // Fill host arrays with device pointers
        for (int i = 0; i < numFrames; i++) {
            h_data_array[i] = frames[i].d_data;
            h_nci_array[i] = peakInfos[i].d_nci;
            h_foldedNci_array[i] = peakInfos[i].d_foldedNci;
            h_noiseEstimation_array[i] = peakInfos[i].d_noiseEstimation;
            h_thresholdingMap_array[i] = peakInfos[i].d_thresholdingMap;
            h_peakList_array[i] = peakInfos[i].d_peakList;
            h_peak_counter_array[i] = peakInfos[i].d_peak_counter;
        }
        
        // Allocate device memory for pointer arrays
        CUDA_CHECK(cudaMalloc(&d_persistent_data_array, numFrames * sizeof(const cuDoubleComplex*)));
        CUDA_CHECK(cudaMalloc(&d_persistent_nci_array, numFrames * sizeof(double*)));
        CUDA_CHECK(cudaMalloc(&d_persistent_foldedNci_array, numFrames * sizeof(double*)));
        CUDA_CHECK(cudaMalloc(&d_persistent_noiseEstimation_array, numFrames * sizeof(double*)));
        CUDA_CHECK(cudaMalloc(&d_persistent_thresholdingMap_array, numFrames * sizeof(double*)));
        CUDA_CHECK(cudaMalloc(&d_persistent_peakList_array, numFrames * sizeof(RadarData::Peak*)));
        CUDA_CHECK(cudaMalloc(&d_persistent_peak_counter_array, numFrames * sizeof(int*)));
        
        // Copy pointer arrays from host to device
        CUDA_CHECK(cudaMemcpy(d_persistent_data_array, h_data_array, numFrames * sizeof(const cuDoubleComplex*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_persistent_nci_array, h_nci_array, numFrames * sizeof(double*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_persistent_foldedNci_array, h_foldedNci_array, numFrames * sizeof(double*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_persistent_noiseEstimation_array, h_noiseEstimation_array, numFrames * sizeof(double*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_persistent_thresholdingMap_array, h_thresholdingMap_array, numFrames * sizeof(double*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_persistent_peakList_array, h_peakList_array, numFrames * sizeof(RadarData::Peak*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_persistent_peak_counter_array, h_peak_counter_array, numFrames * sizeof(int*), cudaMemcpyHostToDevice));
        
        // Free host arrays
        delete[] h_data_array;
        delete[] h_nci_array;
        delete[] h_foldedNci_array;
        delete[] h_noiseEstimation_array;
        delete[] h_thresholdingMap_array;
        delete[] h_peakList_array;
        delete[] h_peak_counter_array;
        
        persistent_arrays_initialized = true;
        persistent_arrays_size = numFrames;
        std::cout << "Persistent device arrays initialized for " << numFrames << " frames" << std::endl;
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in initializePersistentArrays: " << e.what() << std::endl;
        cleanupPersistentArrays();
        return false;
    }
}

/**
 * Clean up persistent device arrays
 */
void cleanupPersistentArrays() {
    if (!persistent_arrays_initialized) {
        return;
    }
    
    try {
        if (d_persistent_data_array) CUDA_CHECK(cudaFree(d_persistent_data_array));
        if (d_persistent_nci_array) CUDA_CHECK(cudaFree(d_persistent_nci_array));
        if (d_persistent_foldedNci_array) CUDA_CHECK(cudaFree(d_persistent_foldedNci_array));
        if (d_persistent_noiseEstimation_array) CUDA_CHECK(cudaFree(d_persistent_noiseEstimation_array));
        if (d_persistent_thresholdingMap_array) CUDA_CHECK(cudaFree(d_persistent_thresholdingMap_array));
        if (d_persistent_peakList_array) CUDA_CHECK(cudaFree(d_persistent_peakList_array));
        if (d_persistent_peak_counter_array) CUDA_CHECK(cudaFree(d_persistent_peak_counter_array));
        
        d_persistent_data_array = nullptr;
        d_persistent_nci_array = nullptr;
        d_persistent_foldedNci_array = nullptr;
        d_persistent_noiseEstimation_array = nullptr;
        d_persistent_thresholdingMap_array = nullptr;
        d_persistent_peakList_array = nullptr;
        d_persistent_peak_counter_array = nullptr;
        
        persistent_arrays_initialized = false;
        persistent_arrays_size = 0;
        std::cout << "Persistent device arrays cleaned up" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in cleanupPersistentArrays: " << e.what() << std::endl;
    }
}

/**
 * Function to perform batch CFAR peak detection on multiple frames
 * 
 * @param frames Array of radar frames to process
 * @param peakInfos Array of peak info structures to store results
 * @param numFrames Number of frames to process
 * @param usePersistentArrays Whether to use persistent device arrays
 */
void batchPeakDetectionPipeline(
    const std::vector<RadarData::Frame>& frames, 
    std::vector<RadarData::peakInfo>& peakInfos,
    int numFrames,
    bool usePersistentArrays
) {
    try {
        std::cout << "Batch Peak Detection: " << numFrames << " frames in parallel" << std::endl;
        
        if (frames.size() < numFrames || peakInfos.size() < numFrames) {
            throw std::runtime_error("frames or peakInfos vector size less than numFrames");
        }
        
        // Get dimensions from first frame
        int num_receivers = frames[0].num_receivers;
        int num_chirps = frames[0].num_chirps;
        int num_samples = frames[0].num_samples;
        int max_num_peaks = peakInfos[0].max_num_peaks;
        
        // Calculate CFAR threshold factor alpha based on false alarm rate
        double alpha = RadarConfig::TRAINING_CELLS * 
                     (std::pow(RadarConfig::FALSE_ALARM_RATE, -1.0 / RadarConfig::TRAINING_CELLS) - 1);
        
        // Reset peak counters for all frames
        for (int i = 0; i < numFrames; i++) {
            CUDA_CHECK(cudaMemset(peakInfos[i].d_peak_counter, 0, sizeof(int)));
        }
        
        // Pointers for device arrays
        const cuDoubleComplex** d_data_array;
        double** d_nci_array;
        double** d_foldedNci_array;
        double** d_noiseEstimation_array;
        double** d_thresholdingMap_array;
        RadarData::Peak** d_peakList_array;
        int** d_peak_counter_array;
        
        // Flag to track if we need to free arrays at the end
        bool need_cleanup = false;
        
        // Check if we should use persistent arrays
        if (usePersistentArrays && persistent_arrays_initialized && persistent_arrays_size >= numFrames) {
            std::cout << "  Using persistent device arrays for batch peak detection" << std::endl;
            
            // Use the persistent arrays
            d_data_array = d_persistent_data_array;
            d_nci_array = d_persistent_nci_array;
            d_foldedNci_array = d_persistent_foldedNci_array;
            d_noiseEstimation_array = d_persistent_noiseEstimation_array;
            d_thresholdingMap_array = d_persistent_thresholdingMap_array;
            d_peakList_array = d_persistent_peakList_array;
            d_peak_counter_array = d_persistent_peak_counter_array;
        }
        else {
            // Create temporary arrays if persistent arrays are not available or not requested
            std::cout << "  Creating temporary device arrays for batch peak detection" << std::endl;
            need_cleanup = true;
            
            // Create host arrays to hold pointers
            cuDoubleComplex** h_data_array = new cuDoubleComplex*[numFrames];
            double** h_nci_array = new double*[numFrames];
            double** h_foldedNci_array = new double*[numFrames];
            double** h_noiseEstimation_array = new double*[numFrames];
            double** h_thresholdingMap_array = new double*[numFrames];
            RadarData::Peak** h_peakList_array = new RadarData::Peak*[numFrames];
            int** h_peak_counter_array = new int*[numFrames];
            
            // Fill host arrays with device pointers
            for (int i = 0; i < numFrames; i++) {
                h_data_array[i] = frames[i].d_data;
                h_nci_array[i] = peakInfos[i].d_nci;
                h_foldedNci_array[i] = peakInfos[i].d_foldedNci;
                h_noiseEstimation_array[i] = peakInfos[i].d_noiseEstimation;
                h_thresholdingMap_array[i] = peakInfos[i].d_thresholdingMap;
                h_peakList_array[i] = peakInfos[i].d_peakList;
                h_peak_counter_array[i] = peakInfos[i].d_peak_counter;
            }
            
            // Allocate device memory for pointer arrays
            CUDA_CHECK(cudaMalloc(&d_data_array, numFrames * sizeof(const cuDoubleComplex*)));
            CUDA_CHECK(cudaMalloc(&d_nci_array, numFrames * sizeof(double*)));
            CUDA_CHECK(cudaMalloc(&d_foldedNci_array, numFrames * sizeof(double*)));
            CUDA_CHECK(cudaMalloc(&d_noiseEstimation_array, numFrames * sizeof(double*)));
            CUDA_CHECK(cudaMalloc(&d_thresholdingMap_array, numFrames * sizeof(double*)));
            CUDA_CHECK(cudaMalloc(&d_peakList_array, numFrames * sizeof(RadarData::Peak*)));
            CUDA_CHECK(cudaMalloc(&d_peak_counter_array, numFrames * sizeof(int*)));
            
            // Copy pointer arrays from host to device
            CUDA_CHECK(cudaMemcpy(d_data_array, h_data_array, numFrames * sizeof(const cuDoubleComplex*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_nci_array, h_nci_array, numFrames * sizeof(double*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_foldedNci_array, h_foldedNci_array, numFrames * sizeof(double*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_noiseEstimation_array, h_noiseEstimation_array, numFrames * sizeof(double*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_thresholdingMap_array, h_thresholdingMap_array, numFrames * sizeof(double*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_peakList_array, h_peakList_array, numFrames * sizeof(RadarData::Peak*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_peak_counter_array, h_peak_counter_array, numFrames * sizeof(int*), cudaMemcpyHostToDevice));
            
            // Free host arrays
            delete[] h_data_array;
            delete[] h_nci_array;
            delete[] h_foldedNci_array;
            delete[] h_noiseEstimation_array;
            delete[] h_thresholdingMap_array;
            delete[] h_peakList_array;
            delete[] h_peak_counter_array;
        }
        
        // Configure kernel execution parameters
        dim3 block(3, 8, 8); // Thread block dimensions
        
        // Calculate grid dimensions for all frames
        dim3 grid(
            ((num_receivers + block.x - 1) / block.x) * numFrames, // Multiply by numFrames
            (num_chirps + block.y - 1) / block.y,
            (num_samples + block.z - 1) / block.z
        );
        
        std::cout << "  Launch parameters: " << grid.x << "x" << grid.y << "x" << grid.z 
                  << " blocks, " << block.x << "x" << block.y << "x" << block.z 
                  << " threads per block" << std::endl;
        
        // Execute batch peak detection kernel
        batchCfarPeakDetectionKernel<<<grid, block>>>(
            d_nci_array,
            d_foldedNci_array,
            d_noiseEstimation_array,
            d_thresholdingMap_array,
            d_peakList_array,
            d_data_array,
            d_peak_counter_array,
            num_receivers,
            num_chirps,
            num_samples,
            alpha,
            max_num_peaks,
            numFrames
        );
        
        // Wait for kernel execution to complete
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error in batchCfarPeakDetectionKernel: ") + 
                                     cudaGetErrorString(err));
        }
        
        // Copy peak counts from device to host
        for (int i = 0; i < numFrames; i++) {
            CUDA_CHECK(cudaMemcpy(&peakInfos[i].num_peaks, peakInfos[i].d_peak_counter, 
                                 sizeof(int), cudaMemcpyDeviceToHost));
                                 
            // Ensure peak count doesn't exceed maximum
            if (peakInfos[i].num_peaks > max_num_peaks) {
                std::cerr << "Warning: Peak count for frame " << i << " exceeds maximum (" 
                          << peakInfos[i].num_peaks << " > " << max_num_peaks 
                          << "). Truncating to maximum." << std::endl;
                peakInfos[i].num_peaks = max_num_peaks;
            }
        }
        
        // Free device memory for pointer arrays only if we created them
        if (need_cleanup) {
            CUDA_CHECK(cudaFree(d_data_array));
            CUDA_CHECK(cudaFree(d_nci_array));
            CUDA_CHECK(cudaFree(d_foldedNci_array));
            CUDA_CHECK(cudaFree(d_noiseEstimation_array));
            CUDA_CHECK(cudaFree(d_thresholdingMap_array));
            CUDA_CHECK(cudaFree(d_peakList_array));
            CUDA_CHECK(cudaFree(d_peak_counter_array));
        }
        
        std::cout << "Batch peak detection completed for " << numFrames << " frames" << std::endl;
        
        // Print peak counts for each frame
        int total_peaks = 0;
        for (int i = 0; i < numFrames; i++) {
            std::cout << "  Frame " << i+1 << ": " << peakInfos[i].num_peaks << " peaks detected" << std::endl;
            total_peaks += peakInfos[i].num_peaks;
        }
        
        std::cout << "  Total peaks detected across all frames: " << total_peaks << std::endl;
        std::cout << "  Average peaks per frame: " << static_cast<double>(total_peaks) / numFrames << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in batchPeakDetectionPipeline: " << e.what() << std::endl;
        throw;
    }
}

} // namespace BatchPeakDetection
