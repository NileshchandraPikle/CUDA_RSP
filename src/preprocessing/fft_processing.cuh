#ifndef FFT_PROCESSING_H
#define FFT_PROCESSING_H

#include "datatypes.cuh"

namespace fftProcessing
{
    __global__ void apply_hilbert_transform_samples(cuDoubleComplex* d_data, int num_receivers, int num_chirps, int num_samples);
    __global__ void apply_fft1(cuDoubleComplex* d_data, int num_receivers, int num_chirps, int num_samples);
    __global__ void apply_fft2(cuDoubleComplex* data, int num_receivers, int num_chirps, int num_samples);
    __device__ void fft(RadarData::Complex* data, size_t length, bool inverse = false);
    void fftProcessPipeline(RadarData::Frame& frame);
    __device__ void apply_hanning_window(RadarData::Complex* data, size_t length);
    __device__ void normalize_fft_output(RadarData::Complex* data, size_t length);
}

#endif