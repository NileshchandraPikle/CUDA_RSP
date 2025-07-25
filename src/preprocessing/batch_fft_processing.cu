#include <vector>
#include <complex>
#include <cuComplex.h>
#include <iostream>
#include <iomanip>
#include "../data_types/datatypes.cuh"
#include "../config/config.hpp"
#include "fft_processing.cuh"
#include "batch_fft_processing.cuh"
#include "../cuda_utils/cuda_utils.hpp"

namespace BatchFFTProcessing {

// Helper: In-place Cooley-Tukey FFT (radix-2, decimation-in-time) for cuDoubleComplex
__device__ void batch_fft(cuDoubleComplex* data, size_t length, bool inverse) {
    if (length <= 1) return;

    // Bit reversal permutation
    size_t j = 0;
    for (size_t i = 0; i < length; ++i) {
        if (i < j) {
            cuDoubleComplex temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
        size_t m = length >> 1;
        while (m && (j & m)) {
            j ^= m;
            m >>= 1;
        }
        j ^= m;
    }

    // FFT
    for (size_t s = 2; s <= length; s <<= 1) {
        double angle = 2 * RadarConfig::PI / s * (inverse ? -1 : 1);
        cuDoubleComplex ws = make_cuDoubleComplex(cos(angle), sin(angle));
        
        for (size_t k = 0; k < length; k += s) {
            cuDoubleComplex w = make_cuDoubleComplex(1.0, 0.0);
            for (size_t m = 0; m < s / 2; ++m) {
                cuDoubleComplex u = data[k + m];
                cuDoubleComplex t = cuCmul(w, data[k + m + s / 2]);
                
                data[k + m] = cuCadd(u, t);
                data[k + m + s / 2] = cuCsub(u, t);
                w = cuCmul(w, ws);
            }
        }
    }

    // Normalize if inverse
    if (inverse) {
        for (size_t i = 0; i < length; ++i){
            data[i].x /= static_cast<double>(length);
            data[i].y /= static_cast<double>(length);
        }
    }
}

__device__ void batch_apply_hanning_window(cuDoubleComplex* data, size_t length) {
    for (size_t n = 0; n < length; ++n) {
        double w = 0.5 * (1 - cos(2 * RadarConfig::PI * n / (length - 1)));
        data[n] = make_cuDoubleComplex(data[n].x * w, data[n].y * w);
    }
}

__device__ void batch_normalize_fft_output(cuDoubleComplex* data, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        data[i].x /= static_cast<double>(length);
        data[i].y /= static_cast<double>(length);
    }
}

// Kernel to apply Hilbert transform to multiple frames
__global__ void batch_apply_hilbert_transform_samples(
    cuDoubleComplex** d_data_array,
    int num_receivers, 
    int num_chirps, 
    int num_samples,
    int num_frames) 
{
    // Calculate global thread ID
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate which frame, receiver, and chirp this thread processes
    int frame_idx = gid / (num_receivers * num_chirps);
    int local_idx = gid % (num_receivers * num_chirps);
    int r = local_idx / num_chirps;
    int c = local_idx % num_chirps;
    
    // Bounds check
    if (frame_idx >= num_frames || r >= num_receivers || c >= num_chirps) return;
    
    // Get pointer to this frame's data
    cuDoubleComplex* frame_data = d_data_array[frame_idx];
    
    // Apply Hilbert transform to this receiver/chirp combination
    int N = num_samples;
    cuDoubleComplex* temp = frame_data + (r * num_chirps + c) * N;
    
    // Perform FFT
    batch_fft(temp, N, false);
    
    // Apply Hilbert filter in frequency domain
    for(int s = 1; s < N / 2; ++s) {
        temp[s] = make_cuDoubleComplex(temp[s].x * 2, temp[s].y * 2);
    }
    
    for(int s = N/2; s < N; ++s) {
        temp[s] = make_cuDoubleComplex(0, 0);
    }
}

// Kernel to apply FFT1 to multiple frames
__global__ void batch_apply_fft1(
    cuDoubleComplex** d_data_array,
    int num_receivers, 
    int num_chirps, 
    int num_samples,
    int num_frames) 
{
    // Calculate global thread ID
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate which frame, receiver, and chirp this thread processes
    int frame_idx = gid / (num_receivers * num_chirps);
    int local_idx = gid % (num_receivers * num_chirps);
    int r = local_idx / num_chirps;
    int c = local_idx % num_chirps;
    
    // Bounds check
    if (frame_idx >= num_frames || r >= num_receivers || c >= num_chirps) return;
    
    // Get pointer to this frame's data
    cuDoubleComplex* frame_data = d_data_array[frame_idx];
    
    // Apply FFT1 to this receiver/chirp combination
    int N = num_samples;
    cuDoubleComplex* data = frame_data + (r * num_chirps + c) * N;
    
    // Apply FFT
    batch_fft(data, num_samples, false);
}

// Kernel to apply FFT2 to multiple frames
__global__ void batch_apply_fft2(
    cuDoubleComplex** d_data_array,
    int num_receivers, 
    int num_chirps, 
    int num_samples,
    int num_frames) 
{
    // Calculate global thread ID
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate which frame, receiver, and sample this thread processes
    int frame_idx = gid / (num_receivers * num_samples);
    int local_idx = gid % (num_receivers * num_samples);
    int r = local_idx / num_samples;
    int s = local_idx % num_samples;
    
    // Bounds check
    if (frame_idx >= num_frames || r >= num_receivers || s >= num_samples) return;
    
    // Get pointer to this frame's data
    cuDoubleComplex* data = d_data_array[frame_idx];
    
    // Temporary array for chirp data
    cuDoubleComplex temp[128]; // Assuming max chirps is 128
    
    // Extract chirp data for this receiver/sample combination
    for (int c = 0; c < num_chirps; ++c) {
        temp[c] = data[(r * num_chirps + c) * num_samples + s];
    }
    
    // Apply Hanning window
    batch_apply_hanning_window(temp, num_chirps);
    
    // Apply FFT
    batch_fft(temp, num_chirps, false);
    
    // Normalize
    batch_normalize_fft_output(temp, num_chirps);
    
    // Write back
    for (int c = 0; c < num_chirps; ++c) {
        data[(r * num_chirps + c) * num_samples + s] = temp[c];
    }
}

// Host function to process multiple frames in batch
void batchFFTProcessPipeline(std::vector<RadarData::Frame>& frames, int numFrames) {
    if (frames.size() < numFrames || numFrames <= 0) {
        throw std::runtime_error("Invalid number of frames for batch processing");
    }
    
    // Create an array of device pointers for kernel invocation
    cuDoubleComplex** h_data_array = new cuDoubleComplex*[numFrames];
    cuDoubleComplex** d_data_array;
    
    // Fill array with device pointers from each frame
    for (int i = 0; i < numFrames; i++) {
        h_data_array[i] = reinterpret_cast<cuDoubleComplex*>(frames[i].d_data);
    }
    
    // Allocate device memory for the array of pointers
    CUDA_CHECK(cudaMalloc(&d_data_array, numFrames * sizeof(cuDoubleComplex*)));
    
    // Copy the array of pointers to device
    CUDA_CHECK(cudaMemcpy(
        d_data_array, 
        h_data_array, 
        numFrames * sizeof(cuDoubleComplex*), 
        cudaMemcpyHostToDevice
    ));
    
    // Get dimensions from the first frame (assuming all frames have same dimensions)
    int num_receivers = frames[0].num_receivers;
    int num_chirps = frames[0].num_chirps;
    int num_samples = frames[0].num_samples;
    
    // Calculate total number of threads needed for each kernel
    int total_threads_fft1 = numFrames * num_receivers * num_chirps;
    int total_threads_fft2 = numFrames * num_receivers * num_samples;
    
    // Define thread block sizes
    int threads_per_block = 256;
    
    // Calculate grid sizes
    int blocks_fft1 = (total_threads_fft1 + threads_per_block - 1) / threads_per_block;
    int blocks_fft2 = (total_threads_fft2 + threads_per_block - 1) / threads_per_block;
    
    std::cout << "Batch FFT Processing: " << numFrames << " frames in parallel" << std::endl;
    std::cout << "  Launch parameters for Hilbert & FFT1: " << blocks_fft1 
              << " blocks, " << threads_per_block << " threads per block" 
              << " (total: " << total_threads_fft1 << " threads)" << std::endl;
    std::cout << "  Launch parameters for FFT2: " << blocks_fft2 
              << " blocks, " << threads_per_block << " threads per block"
              << " (total: " << total_threads_fft2 << " threads)" << std::endl;
    
    // Apply Hilbert transform to all frames in batch
    batch_apply_hilbert_transform_samples<<<blocks_fft1, threads_per_block>>>(
        d_data_array,
        num_receivers,
        num_chirps,
        num_samples,
        numFrames
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Apply FFT1 to all frames in batch
    batch_apply_fft1<<<blocks_fft1, threads_per_block>>>(
        d_data_array,
        num_receivers,
        num_chirps,
        num_samples,
        numFrames
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Apply FFT2 to all frames in batch
    batch_apply_fft2<<<blocks_fft2, threads_per_block>>>(
        d_data_array,
        num_receivers,
        num_chirps,
        num_samples,
        numFrames
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Clean up
    CUDA_CHECK(cudaFree(d_data_array));
    delete[] h_data_array;
}

} // namespace BatchFFTProcessing
