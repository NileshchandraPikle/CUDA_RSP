#include <iostream>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <vector>
#include "config/config.hpp"
#include "data_types/datatypes.cuh"
#include "preprocessing/fft_processing.cuh"
#include "preprocessing/batch_fft_processing.cuh"
#include "peak_detection/peak_detection.cuh"
#include "peak_detection/batch_peak_detection.cuh"
#include "mimo_synthesis/mimo_synthesis.cuh"
#include "mimo_synthesis/batch_mimo_synthesis.cuh"
#include "doa_processing/doa_processing.cuh"
#include "target_processing/target_processing.cuh"
#include "rcs/rcs.cuh"
#include "ego_estimation/ego_estimation.cuh"
#include "ghost_removal/ghost_removal.cuh"

/**
 * CUDA Radar Signal Processing Pipeline with Batch Processing
 * 
 * This implementation processes multiple radar data frames in parallel using CUDA.
 * The pipeline includes:
 * 1. FFT processing (Batch implementation)
 * 2. Peak detection (Batch implementation)
 * 3. MIMO synthesis (Batch implementation)
 * 4. Direction of Arrival processing (Commented out)
 * 5. Target detection (Commented out)
 * 6. RCS estimation (Commented out)
 * 7. Ego motion estimation (Commented out)
 * 8. Ghost target removal (Commented out)
 * 
 * This version implements batch processing for multiple stages,
 * processing multiple frames simultaneously in parallel.
 */


/**
 * Helper function to print processing step timing information
 */
void printTimingInfo(const std::string& stepName, const std::chrono::duration<double>& elapsed) {
    std::cout << "Time taken for " << std::left << std::setw(25) << stepName << ": "
              << std::fixed << std::setprecision(6) << elapsed.count() << " seconds" << std::endl;
}

/**
 * Main function implementing the radar signal processing pipeline with batch processing
 */
int main() 
{
    try {
        // Load radar configuration
        std::cout << "Loading radar configuration..." << std::endl;
        RadarConfig::Config rconfig = RadarConfig::load_config();
        std::cout << "Radar Configuration Loaded successfully" << std::endl;
        
        // Initialize batch processing parameters
        constexpr int BATCH_SIZE = 20;  // Number of frames to process in parallel
        bool run_verification = true;   // Set to true to verify batch processing against single-frame
                
        // Initialize frame and peak info data structures for a single frame
        // (used only for reference and comparison)
        RadarData::Frame frame(rconfig.num_receivers, rconfig.num_chirps, rconfig.num_samples);
        RadarData::peakInfo peakinfo(rconfig.num_receivers, rconfig.num_chirps, rconfig.num_samples);

        // Initialize maximum-capacity structures once outside the loop
        const int MAX_EXPECTED_PEAKS = peakinfo.max_num_peaks;
        
        // Pre-allocate DoA structure with maximum capacity
        RadarData::DoAInfo doaInfo(MAX_EXPECTED_PEAKS, rconfig.num_receivers);
        doaInfo.initialize();
        
        // Pre-allocate target results with maximum capacity
        RadarData::TargetResults targetResults(MAX_EXPECTED_PEAKS);
        
        // Pre-allocate filtered results (for ghost removal) with maximum capacity
        RadarData::TargetResults filteredResults(MAX_EXPECTED_PEAKS);
        filteredResults.free_device(); // Only need host memory for filtering
        // Batch processing code
        // Vector to store all frames for batch processing
        std::vector<RadarData::Frame> frames;
        // Initialize batch frames
        size_t total_memory = RadarData::initializeBatchFrames(
            frames,
            BATCH_SIZE,
            rconfig.num_receivers,
            rconfig.num_chirps,
            rconfig.num_samples
        );
        size_t frame_size = RadarData::frame_size_bytes(frames[0]);
        int total_threads_fft2 = BATCH_SIZE * rconfig.num_receivers * rconfig.num_samples;
        //*********************STEP 1 FFT PROCESSING *******************
        auto start = std::chrono::high_resolution_clock::now();
        BatchFFTProcessing::batchFFTProcessPipeline(frames, BATCH_SIZE);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        printTimingInfo("Batch FFT", elapsed);
        double batch_avg = elapsed.count() / BATCH_SIZE;
        int frames_processed = BATCH_SIZE;
        //*********************STEP 2 PEAK DETECTION *******************
        std::vector<RadarData::peakInfo> peakInfos;
        peakInfos.reserve(frames_processed);
        for (int i = 0; i < frames_processed; ++i) {
            peakInfos.emplace_back(rconfig.num_receivers, rconfig.num_chirps, rconfig.num_samples);
        }
        bool use_persistent_arrays = false;
        if (!BatchPeakDetection::persistent_arrays_initialized) {
            if (BatchPeakDetection::initializePersistentArrays(frames, peakInfos, frames_processed)) {
                use_persistent_arrays = true;
            }
        } else {
            use_persistent_arrays = true;
        }
        start = std::chrono::high_resolution_clock::now();
        BatchPeakDetection::batchPeakDetectionPipeline(frames, peakInfos, frames_processed, use_persistent_arrays);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        printTimingInfo("Batch Peak Detection", elapsed);
        double batch_peak_avg = elapsed.count() / frames_processed;
        int total_peaks = 0;
        for (int i = 0; i < frames_processed; ++i) {
            total_peaks += peakInfos[i].num_peaks;
        }
        double avg_peaks_per_frame = static_cast<double>(total_peaks) / frames_processed;
        //*********************STEP 3 MIMO SYNTHESIS *******************
        bool use_mimo_persistent_arrays = false;
        if (!BatchMIMOSynthesis::persistent_arrays_initialized) {
            if (BatchMIMOSynthesis::initializePersistentArrays(frames, peakInfos, frames_processed)) {
                use_mimo_persistent_arrays = true;
            }
        } else {
            use_mimo_persistent_arrays = true;
        }
        start = std::chrono::high_resolution_clock::now();
        BatchMIMOSynthesis::batchSynthesizePeaks(frames, peakInfos, frames_processed);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        printTimingInfo("Batch MIMO Synthesis", elapsed);
        double batch_mimo_avg = elapsed.count() / frames_processed;
        // Calculate and print overall processing time
        double total_processing_time = 0.0;
        int frames_count = 0;
        total_processing_time += batch_avg * frames_processed;
        total_processing_time += batch_peak_avg * frames_processed;
        total_processing_time += batch_mimo_avg * frames_processed;
        frames_count = frames_processed;
        std::cout << "Total batch pipeline time: " << std::fixed << std::setprecision(6) << total_processing_time << " seconds\n";
        // Clean up MIMO synthesis persistent arrays if initialized
        if (BatchMIMOSynthesis::persistent_arrays_initialized) {
            BatchMIMOSynthesis::cleanupPersistentArrays();
        }
        RadarData::cleanupBatchResources(
            frames,
            peakInfos,
            doaInfo,
            targetResults,
            BatchPeakDetection::persistent_arrays_initialized,
            &BatchPeakDetection::cleanupPersistentArrays
        );
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "\nERROR: Unknown exception occurred" << std::endl;
        return 2;
    }
}
