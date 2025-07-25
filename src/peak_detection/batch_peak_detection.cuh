#ifndef BATCH_PEAK_DETECTION_HPP
#define BATCH_PEAK_DETECTION_HPP

#include "../data_types/datatypes.cuh"
#include <vector>

namespace BatchPeakDetection {
    // Persistent device pointer arrays for batch processing
    extern const cuDoubleComplex** d_persistent_data_array;
    extern double** d_persistent_nci_array;
    extern double** d_persistent_foldedNci_array;
    extern double** d_persistent_noiseEstimation_array;
    extern double** d_persistent_thresholdingMap_array;
    extern RadarData::Peak** d_persistent_peakList_array;
    extern int** d_persistent_peak_counter_array;
    extern bool persistent_arrays_initialized;
    extern int persistent_arrays_size;
    
    /**
     * @brief Initialize persistent device arrays for batch processing
     * 
     * @param frames Vector of Frame objects to process in parallel
     * @param peakInfos Vector of peakInfo objects to store results
     * @param numFrames Number of frames to process
     * @return true if initialization successful
     */
    bool initializePersistentArrays(
        const std::vector<RadarData::Frame>& frames,
        const std::vector<RadarData::peakInfo>& peakInfos,
        int numFrames
    );
    
    /**
     * @brief Clean up persistent device arrays
     */
    void cleanupPersistentArrays();

    /**
     * @brief Process multiple frames in batch mode for peak detection using a single kernel launch
     *
     * @param frames Vector of Frame objects to process in parallel
     * @param peakInfos Vector of peakInfo objects to store results
     * @param numFrames Number of frames to process
     * @param usePersistentArrays Whether to use persistent device arrays (avoids CPU-to-GPU transfers)
     */
    void batchPeakDetectionPipeline(
        const std::vector<RadarData::Frame>& frames, 
        std::vector<RadarData::peakInfo>& peakInfos,
        int numFrames,
        bool usePersistentArrays = false
    );

    /**
     * @brief CUDA kernel for batch CFAR peak detection
     * 
     * This kernel performs Constant False Alarm Rate (CFAR) detection on multiple radar data frames.
     * Each thread block processes one frame, and each thread processes one cell in that frame.
     * 
     * @param d_nci_array Array of noise cell information pointers
     * @param d_foldedNci_array Array of folded noise cell information pointers
     * @param d_noiseEstimation_array Array of noise estimation pointers
     * @param d_thresholdingMap_array Array of thresholding map pointers
     * @param d_peakList_array Array of peak list pointers
     * @param d_data_array Array of radar data pointers
     * @param d_peak_counter_array Array of peak counter pointers
     * @param num_receivers Number of receivers
     * @param num_chirps Number of chirps
     * @param num_samples Number of samples per chirp
     * @param alpha CFAR threshold factor
     * @param max_num_peaks Maximum number of peaks that can be stored
     * @param numFrames Number of frames to process
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
    );
}

#endif // BATCH_PEAK_DETECTION_HPP
