#include <vector>
#include <complex>
#include <cuComplex.h>
#include <iostream>
#include "../data_types/datatypes.cuh"
#include "../config/config.hpp"
#include "fft_processing.cuh"
#include "../cuda_utils/cuda_utils.hpp"


namespace fftProcessing {

// Helper: In-place Cooley-Tukey FFT (radix-2, decimation-in-time)
__device__ void fft(cuDoubleComplex* data, size_t length, bool inverse) {
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

__device__ void apply_hanning_window(cuDoubleComplex* data, size_t length) {
    for (size_t n = 0; n < length; ++n) {
        double w = 0.5 * (1 - cos(2 * RadarConfig::PI * n / (length - 1)));
        data[n] = make_cuDoubleComplex(data[n].x * w, data[n].y * w);
    }
}


__device__ void normalize_fft_output(cuDoubleComplex* data, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        data[i].x /= static_cast<double>(length);
        data[i].y /= static_cast<double>(length);
    }
}


__global__ void apply_hilbert_transform_samples(cuDoubleComplex* d_data, int num_receivers, int num_chirps, int num_samples) {
    // For each receiver/chirp, apply Hilbert transform to the samples
     
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_receivers * num_chirps) return;
    int r = gid / num_chirps;
    int c = gid % num_chirps;
    int N = num_samples;
    
    cuDoubleComplex *temp = d_data + (r * num_chirps + c) * N;

    fft(temp, N, false);

    // Apply Hilbert filter in frequency domain
     for(int s = 1; s < N / 2; ++s) {
         temp[s] = make_cuDoubleComplex(temp[s].x * 2, temp[s].y * 2);
     }

     for(int s = N/2; s < N; ++s) {
         temp[s] = make_cuDoubleComplex(0, 0);
     }

}

__global__ void apply_fft1(cuDoubleComplex* d_data, int num_receivers, int num_chirps, int num_samples) {
    // FFT along the samples dimension for each receiver/chirp
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_receivers * num_chirps) return;
    int r = gid / num_chirps;
    int c = gid % num_chirps;
    int N = num_samples;

    cuDoubleComplex *data = d_data + (r * num_chirps + c) * N;
    // Apply Hanning window

            //apply_hanning_window(data,num_samples);
            fft(data, num_samples, false);
            //normalize_fft_output(data,num_samples);     
}


__global__ void apply_fft2(cuDoubleComplex* data, int num_receivers, int num_chirps, int num_samples) {
    // FFT along the chirps dimension for each receiver/sample

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_receivers * num_samples) return;
    int r = gid / num_samples;
    int s = gid % num_samples;
   
    cuDoubleComplex temp[128];
    for (int c = 0; c < num_chirps; ++c) {
        temp[c] = data[(r * num_chirps + c) * num_samples + s];
    }
           apply_hanning_window(temp,num_chirps);
           fft(temp, num_chirps, false);
           normalize_fft_output(temp, num_chirps);
    for (int c = 0; c < num_chirps; ++c) {
        data[(r * num_chirps + c) * num_samples + s] = temp[c];
    }
}



void fftProcessPipeline(RadarData::Frame& frame) {

    int threadsfft1 = 256;
    int blocksfft1  = (frame.num_receivers * frame.num_chirps + threadsfft1 - 1) / threadsfft1;

   apply_hilbert_transform_samples<<<blocksfft1, threadsfft1>>>(
        reinterpret_cast<cuDoubleComplex*>(frame.d_data),
        frame.num_receivers, frame.num_chirps, frame.num_samples
    );
 CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    
    apply_fft1<<<blocksfft1, threadsfft1>>>(reinterpret_cast<cuDoubleComplex*>(frame.d_data), frame.num_receivers, frame.num_chirps, frame.num_samples);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    
    int threadsfft2 = 256;
    int blocksfft2 = (frame.num_receivers * frame.num_samples + threadsfft2 - 1) / threadsfft2;

    apply_fft2<<<blocksfft2, threadsfft2>>>(reinterpret_cast<cuDoubleComplex*>(frame.d_data), frame.num_receivers, frame.num_chirps, frame.num_samples);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
} 
}
