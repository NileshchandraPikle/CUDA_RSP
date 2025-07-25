#ifndef BATCH_FFT_PROCESSING_H
#define BATCH_FFT_PROCESSING_H

#include "../data_types/datatypes.cuh"
#include <vector>
#include <iostream>
#include "../config/config.hpp"

namespace BatchFFTProcessing
{

    /**
     * @brief Process multiple frames in batch mode using a single kernel launch
     *
     * @param frames Vector of Frame objects to process in parallel
     * @param numFrames Number of frames to process
     */
    void batchFFTProcessPipeline(std::vector<RadarData::Frame>& frames, int numFrames);

    /**
     * @brief Apply Hilbert transform to multiple frames in batch
     * 
     * @param d_data_array Device pointers to frames data
     * @param num_receivers Number of receivers
     * @param num_chirps Number of chirps
     * @param num_samples Number of samples
     * @param num_frames Number of frames to process in batch
     */
    __global__ void batch_apply_hilbert_transform_samples(
        cuDoubleComplex** d_data_array,
        int num_receivers, 
        int num_chirps, 
        int num_samples,
        int num_frames);

    /**
     * @brief Apply FFT1 to multiple frames in batch
     * 
     * @param d_data_array Device pointers to frames data
     * @param num_receivers Number of receivers
     * @param num_chirps Number of chirps
     * @param num_samples Number of samples
     * @param num_frames Number of frames to process in batch
     */
    __global__ void batch_apply_fft1(
        cuDoubleComplex** d_data_array,
        int num_receivers, 
        int num_chirps, 
        int num_samples,
        int num_frames);

    /**
     * @brief Apply FFT2 to multiple frames in batch
     * 
     * @param d_data_array Device pointers to frames data
     * @param num_receivers Number of receivers
     * @param num_chirps Number of chirps
     * @param num_samples Number of samples
     * @param num_frames Number of frames to process in batch
     */
    __global__ void batch_apply_fft2(
        cuDoubleComplex** d_data_array,
        int num_receivers, 
        int num_chirps, 
        int num_samples,
        int num_frames);
}

#endif
