#ifndef BATCH_MIMO_SYNTHESIS_HPP
#define BATCH_MIMO_SYNTHESIS_HPP

#include <vector>
#include "../data_types/datatypes.cuh"
#include "../cuda_utils/cuda_utils.hpp"

namespace BatchMIMOSynthesis {
    /**
     * @brief Perform batch MIMO synthesis on radar peaks for multiple frames
     *
     * This function extracts signal data at peak locations from all receivers
     * to create virtual array snapshots for direction finding across multiple frames in parallel.
     *
     * @param frames Vector of input radar frames containing signal data
     * @param peakInfos Vector of peak information structures for input peaks and output synthesized data
     * @param batchSize Number of frames to process in parallel
     * @throws std::runtime_error If CUDA operations fail
     */
    void batchSynthesizePeaks(
        const std::vector<RadarData::Frame>& frames, 
        std::vector<RadarData::peakInfo>& peakInfos,
        int batchSize
    );

    /**
     * @brief CUDA kernel to perform batch MIMO synthesis
     *
     * For each peak in multiple frames, this kernel extracts the signal value at the peak location
     * from all receivers to create a virtual array snapshot.
     *
     * @param d_data_batch Array of pointers to frame data in device memory
     * @param d_peakList_batch Array of pointers to peak lists in device memory
     * @param d_peaksnaps_batch Array of pointers to output synthesized peaks in device memory
     * @param num_peaks_batch Array of peak counts for each frame
     * @param frame_index Frame index within the batch
     * @param num_receivers Number of receivers in the array
     * @param num_chirps Number of chirps per frame
     * @param num_samples Number of samples per chirp
     * @param max_num_peaks Maximum number of peaks (buffer size)
     */
    __global__ void batch_synthesize_peaks_kernel(
        cuDoubleComplex** d_data_batch,
        RadarData::Peak** d_peakList_batch,
        cuDoubleComplex** d_peaksnaps_batch,
        const int* num_peaks_batch,
        int frame_index,
        int num_receivers,
        int num_chirps,
        int num_samples,
        int max_num_peaks
    );
    
    /**
     * @brief Initialize persistent arrays for batch MIMO synthesis
     *
     * This function allocates memory for persistent arrays used in batch MIMO synthesis,
     * which can be reused across multiple calls for better performance.
     *
     * @param frames Vector of frames to be processed
     * @param peakInfos Vector of peak information structures
     * @param batchSize Number of frames to process in parallel
     * @return True if initialization succeeded, false otherwise
     */
    bool initializePersistentArrays(
        const std::vector<RadarData::Frame>& frames,
        const std::vector<RadarData::peakInfo>& peakInfos,
        int batchSize
    );
    
    /**
     * @brief Clean up persistent arrays used in batch MIMO synthesis
     *
     * This function frees memory allocated for persistent arrays.
     */
    void cleanupPersistentArrays();
    
    /**
     * @brief Verify batch MIMO synthesis results against single-frame processing
     *
     * This function compares batch MIMO synthesis results with single-frame results
     * to ensure both implementations produce equivalent outputs.
     *
     * @param frames Vector of input radar frames containing signal data
     * @param batchPeakInfos Vector of peak information structures from batch processing
     * @param batchSize Number of frames to verify
     * @param print_details Whether to print detailed comparison results
     * @return Number of frames with mismatches (0 means all match)
     */
    int verifyBatchResults(
        const std::vector<RadarData::Frame>& frames,
        std::vector<RadarData::peakInfo>& batchPeakInfos,
        int batchSize,
        bool print_details = true
    );

    // Flag to track if persistent arrays have been initialized
    extern bool persistent_arrays_initialized;
}
#endif // BATCH_MIMO_SYNTHESIS_HPP
