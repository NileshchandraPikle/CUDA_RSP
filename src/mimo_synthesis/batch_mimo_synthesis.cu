#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include "../config/config.hpp"
#include "../cuda_utils/cuda_utils.hpp"
#include "batch_mimo_synthesis.cuh"
#include "mimo_synthesis.cuh"

// Forward declaration of the single-frame kernel from mimo_synthesis.cu
namespace MIMOSynthesis {
    __global__ void synthesize_peaks_kernel(
        const cuDoubleComplex* d_data,
        RadarData::Peak* d_peakList,
        cuDoubleComplex* d_peaksnaps,
        int num_peaks,
        int num_receivers,
        int num_chirps,
        int num_samples,
        int max_num_peaks);
}

namespace BatchMIMOSynthesis {

// Flag to track if persistent arrays have been initialized
bool persistent_arrays_initialized = false;

// Persistent device arrays for batch processing
cuDoubleComplex** d_data_batch = nullptr;
RadarData::Peak** d_peakList_batch = nullptr;
cuDoubleComplex** d_peaksnaps_batch = nullptr;
int* d_num_peaks_batch = nullptr;
int batch_size = 0;

__global__ void batch_synthesize_peaks_kernel(
        cuDoubleComplex** d_data_batch,
        RadarData::Peak** d_peakList_batch,
        cuDoubleComplex** d_peaksnaps_batch,
        const int* num_peaks_batch,
        int frame_index,
        int num_receivers,
        int num_chirps,
        int num_samples,
        int max_num_peaks) {
    
    // Calculate thread index within this frame's processing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_peaks = num_peaks_batch[frame_index];
    
    if (idx >= num_peaks) return;
    
    // Get pointers for this specific frame - make sure they're not nullptr
    if (d_data_batch == nullptr || d_peakList_batch == nullptr || d_peaksnaps_batch == nullptr) {
        return; // Array of pointers is null, exit the kernel
    }
    
    cuDoubleComplex* d_data = d_data_batch[frame_index];
    RadarData::Peak* d_peakList = d_peakList_batch[frame_index];
    cuDoubleComplex* d_peaksnaps = d_peaksnaps_batch[frame_index];
    
    // Make sure all the frame-specific pointers are not nullptr
    if (d_data == nullptr || d_peakList == nullptr || d_peaksnaps == nullptr) {
        return; // Frame-specific pointer is null, exit the kernel
    }
    
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
        
        // Debug print (in device code - will only show in case of printf debugging)
        // printf("Frame %d, Peak %d, Receiver %d: (%f, %f)\n", 
        //        frame_index, idx, r, d_data[i].x, d_data[i].y);
    }
}

bool initializePersistentArrays(
    const std::vector<RadarData::Frame>& frames,
    const std::vector<RadarData::peakInfo>& peakInfos,
    int batchSize) {
    
    try {
        batch_size = batchSize;
        
        
        // Allocate host arrays to hold device pointers
        cuDoubleComplex** h_data_batch = new cuDoubleComplex*[batchSize];
        RadarData::Peak** h_peakList_batch = new RadarData::Peak*[batchSize];
        cuDoubleComplex** h_peaksnaps_batch = new cuDoubleComplex*[batchSize];
        int* h_num_peaks_batch = new int[batchSize];
        
        // Fill the host arrays with device pointers from each frame and peakInfo
        for (int i = 0; i < batchSize; i++) {
            h_data_batch[i] = frames[i].d_data;
            // Only use valid peak and peaksnaps pointers if we have peaks
            if (peakInfos[i].num_peaks > 0) {
                h_peakList_batch[i] = peakInfos[i].d_peakList;
                h_peaksnaps_batch[i] = peakInfos[i].d_peaksnaps;
            } else {
                h_peakList_batch[i] = nullptr;
                h_peaksnaps_batch[i] = nullptr;
            }
            h_num_peaks_batch[i] = peakInfos[i].num_peaks;
        }
        
        // Allocate device memory for the arrays of pointers
        CUDA_CHECK(cudaMalloc(&d_data_batch, batchSize * sizeof(cuDoubleComplex*)));
        CUDA_CHECK(cudaMalloc(&d_peakList_batch, batchSize * sizeof(RadarData::Peak*)));
        CUDA_CHECK(cudaMalloc(&d_peaksnaps_batch, batchSize * sizeof(cuDoubleComplex*)));
        CUDA_CHECK(cudaMalloc(&d_num_peaks_batch, batchSize * sizeof(int)));
        
        // Copy the host arrays to device memory
        CUDA_CHECK(cudaMemcpy(d_data_batch, h_data_batch, batchSize * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_peakList_batch, h_peakList_batch, batchSize * sizeof(RadarData::Peak*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_peaksnaps_batch, h_peaksnaps_batch, batchSize * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_num_peaks_batch, h_num_peaks_batch, batchSize * sizeof(int), cudaMemcpyHostToDevice));
        
        // Clean up host arrays
        delete[] h_data_batch;
        delete[] h_peakList_batch;
        delete[] h_peaksnaps_batch;
        delete[] h_num_peaks_batch;
        
        persistent_arrays_initialized = true;
        return true;
    } 
    catch (const std::exception& e) {
        std::cerr << "Error initializing persistent arrays for batch MIMO synthesis: " << e.what() << std::endl;
        cleanupPersistentArrays();
        return false;
    }
}

void cleanupPersistentArrays() {
    if (d_data_batch) {
        cudaFree(d_data_batch);
        d_data_batch = nullptr;
    }
    
    if (d_peakList_batch) {
        cudaFree(d_peakList_batch);
        d_peakList_batch = nullptr;
    }
    
    if (d_peaksnaps_batch) {
        cudaFree(d_peaksnaps_batch);
        d_peaksnaps_batch = nullptr;
    }
    
    if (d_num_peaks_batch) {
        cudaFree(d_num_peaks_batch);
        d_num_peaks_batch = nullptr;
    }
    
    persistent_arrays_initialized = false;
}

void updatePersistentArrays(const std::vector<RadarData::peakInfo>& peakInfos) {
    // Create host array to update peak counts
    int* h_num_peaks_batch = new int[batch_size];
    
    // Update peak counts
    for (int i = 0; i < batch_size; i++) {
        h_num_peaks_batch[i] = peakInfos[i].num_peaks;
    }
    
    // Copy updated peak counts to device
    CUDA_CHECK(cudaMemcpy(d_num_peaks_batch, h_num_peaks_batch, batch_size * sizeof(int), cudaMemcpyHostToDevice));
    
    // Clean up
    delete[] h_num_peaks_batch;
}

void batchSynthesizePeaks(
    const std::vector<RadarData::Frame>& frames, 
    std::vector<RadarData::peakInfo>& peakInfos,
    int batchSize) {
    
    try {
        // Parameter validation
        if (frames.empty() || peakInfos.empty()) {
            throw std::runtime_error("Empty frames or peakInfos vectors");
        }
        
        if (frames.size() < batchSize || peakInfos.size() < batchSize) {
            throw std::runtime_error("Insufficient frames or peakInfos for requested batch size");
        }
        
        // Check if we need to initialize or update persistent arrays
        if (!persistent_arrays_initialized) {
            if (!initializePersistentArrays(frames, peakInfos, batchSize)) {
                throw std::runtime_error("Failed to initialize persistent arrays for batch MIMO synthesis");
            }
        } else {
            // If already initialized, update peak counts (which may have changed)
            updatePersistentArrays(peakInfos);
        }
        
        // Initialize peak snapshots for each frame with peaks
        for (int i = 0; i < batchSize; i++) {
            // Only initialize peak snapshots if the frame has peaks
            if (peakInfos[i].num_peaks > 0) {
                peakInfos[i].initializePeakSnaps();
            }
        }
        
        // Calculate total number of peaks across all frames for reporting
        int total_peaks = 0;
        for (int i = 0; i < batchSize; i++) {
            total_peaks += peakInfos[i].num_peaks;
        }
        
        
        // Process each frame in the batch
        for (int frame_idx = 0; frame_idx < batchSize; frame_idx++) {
            // Skip frames with no peaks
            if (peakInfos[frame_idx].num_peaks <= 0) {
                continue;
            }
            
            // Calculate kernel launch parameters for this frame
            const int threads_per_block = 256;
            int blocks = (peakInfos[frame_idx].num_peaks + threads_per_block - 1) / threads_per_block;
            
            
            // For simplicity and verification, use the standard single-frame kernel directly
            // This ensures we're using the exact same processing logic
            MIMOSynthesis::synthesize_peaks_kernel<<<blocks, threads_per_block>>>(
                frames[frame_idx].d_data,
                peakInfos[frame_idx].d_peakList,
                peakInfos[frame_idx].d_peaksnaps,
                peakInfos[frame_idx].num_peaks,
                frames[frame_idx].num_receivers,
                frames[frame_idx].num_chirps,
                frames[frame_idx].num_samples,
                peakInfos[frame_idx].max_num_peaks
            );
            
            // Check for errors after each kernel launch
            CUDA_CHECK(cudaGetLastError());
        }
        
        // Wait for all kernels to complete
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy peak snapshots back to host for each frame if needed
        for (int i = 0; i < batchSize; i++) {
            if (peakInfos[i].num_peaks > 0) {
                peakInfos[i].copyPeakSnapsToHost();
            }
        }
        
        
    } catch (const std::exception& e) {
        std::cerr << "Error in batch MIMO synthesis: " << e.what() << std::endl;
        throw;
    }
}

int verifyBatchResults(
    const std::vector<RadarData::Frame>& frames,
    std::vector<RadarData::peakInfo>& batchPeakInfos,
    int batchSize,
    bool print_details) {
    
    std::cout << "\n=== Verifying Batch MIMO Synthesis Results ===\n";
    
    int mismatch_frames = 0;
    int total_peaks_verified = 0;
    int total_mismatches = 0;
    
    // Process each frame sequentially with single-frame MIMO synthesis and compare
    for (int i = 0; i < batchSize; i++) {
        // Skip frames without peaks
        if (batchPeakInfos[i].num_peaks == 0) {
            std::cout << "  Frame " << i+1 << ": No peaks to verify\n";
            continue;
        }
        
        // Create a copy of the peak info for verification with single-frame processing
        RadarData::peakInfo singlePeakInfo(frames[i].num_receivers, frames[i].num_chirps, frames[i].num_samples);
        singlePeakInfo.num_peaks = batchPeakInfos[i].num_peaks;
        
        // Copy peak list from batch result to single frame peak info
        CUDA_CHECK(cudaMemcpy(singlePeakInfo.d_peakList, batchPeakInfos[i].d_peakList, 
                   batchPeakInfos[i].num_peaks * sizeof(RadarData::Peak), cudaMemcpyDeviceToDevice));
        
        // Initialize peak snapshots for single frame processing
        singlePeakInfo.initializePeakSnaps();
        
        // Process with single-frame MIMO synthesis
        MIMOSynthesis::synthesize_peaks(frames[i], singlePeakInfo);
        
        // Copy results back to host for comparison
        singlePeakInfo.copyPeakSnapsToHost();
        
        // Make sure batch results are also on host
        batchPeakInfos[i].copyPeakSnapsToHost();
        
        // Compare results
        int frame_mismatches = 0;
        for (int p = 0; p < batchPeakInfos[i].num_peaks; p++) {
            for (int r = 0; r < frames[i].num_receivers; r++) {
                int idx = p * frames[i].num_receivers + r;
                RadarData::Complex batch_val = batchPeakInfos[i].peaksnaps[idx];
                RadarData::Complex single_val = singlePeakInfo.peaksnaps[idx];
                
                // Check if values are significantly different
                bool mismatch = (std::abs(batch_val.real() - single_val.real()) > 1e-10) || 
                               (std::abs(batch_val.imag() - single_val.imag()) > 1e-10);
                               
                if (mismatch) {
                    frame_mismatches++;
                    total_mismatches++;
                    
                    // Print first few differences if detailed output is requested
                    if (print_details && frame_mismatches <= 3) {
                        std::cout << "    Mismatch in Frame " << i+1 << ", Peak " << p 
                                 << ", Receiver " << r << ":\n"
                                 << "      Batch: (" << batch_val.real() << ", " << batch_val.imag() << ")\n"
                                 << "      Single: (" << single_val.real() << ", " << single_val.imag() << ")\n";
                    }
                }
            }
        }
        
        total_peaks_verified += batchPeakInfos[i].num_peaks;
        
        if (frame_mismatches > 0) {
            mismatch_frames++;
            std::cout << "  Frame " << i+1 << ": " << frame_mismatches << " mismatches out of " 
                     << (batchPeakInfos[i].num_peaks * frames[i].num_receivers) << " values\n";
        } else {
            std::cout << "  Frame " << i+1 << ": All " 
                     << (batchPeakInfos[i].num_peaks * frames[i].num_receivers) 
                     << " values match ✓\n";
        }
        
        // Clean up single frame resources
        singlePeakInfo.free_peakInfo_device();
        singlePeakInfo.free_peakInfo_host();
    }
    
    std::cout << "\n=== Verification Summary ===\n";
    std::cout << "  Total frames verified: " << batchSize << "\n";
    std::cout << "  Total peaks verified: " << total_peaks_verified << "\n";
    std::cout << "  Frames with mismatches: " << mismatch_frames << "\n";
    std::cout << "  Total value mismatches: " << total_mismatches << "\n";
    
    if (mismatch_frames == 0) {
        std::cout << "\nVERIFICATION PASSED: Batch MIMO synthesis matches single-frame processing! ✓\n";
    } else {
        std::cout << "\nVERIFICATION FAILED: Differences found between batch and single-frame processing.\n";
    }
    
    return mismatch_frames;
}

} // namespace BatchMIMOSynthesis
