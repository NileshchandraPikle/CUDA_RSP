
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
 * 3. MIMO synthesis (Commented out)
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
        std::cout << "\n=== Using BATCH PROCESSING mode ===\n" << std::endl;
        
        // Vector to store all frames for batch processing
        std::vector<RadarData::Frame> frames;
        
        // Initialize batch frames using the dedicated function
        size_t total_memory = RadarData::initializeBatchFrames(
            frames,
            BATCH_SIZE,
            rconfig.num_receivers,
            rconfig.num_chirps,
            rconfig.num_samples
        );
        
        // Calculate frame size in bytes for a single frame
        size_t frame_size = RadarData::frame_size_bytes(frames[0]);
        std::cout << "Single frame size: " << frame_size << " bytes" << std::endl;
        std::cout << "Total batch size: " << (frame_size * BATCH_SIZE) / (1024*1024) << " MB" << std::endl;
        
        // Calculate thread counts for summary (used for reporting only)
        int total_threads_fft2 = BATCH_SIZE * rconfig.num_receivers * rconfig.num_samples;
        
        //*********************STEP 1 FFT PROCESSING *******************
        std::cout << "\nExecuting Batch FFT Processing on " << BATCH_SIZE << " frames..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        // Execute FFT pipeline in batch mode
        BatchFFTProcessing::batchFFTProcessPipeline(frames, BATCH_SIZE);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        printTimingInfo("Batch FFT Processing (" + std::to_string(BATCH_SIZE) + " frames)", elapsed);
        double batch_avg = elapsed.count() / BATCH_SIZE;
        std::cout << "Average per frame: " << batch_avg << " seconds" << std::endl;
        
        // Print summary of the FFT processing with standardized format
        int frames_processed = BATCH_SIZE;
        double processing_time = batch_avg;

        std::cout << "\n======= FFT Processing Summary =======" << std::endl;
        std::cout << "Processing mode: BATCH (parallel)" << std::endl;
        std::cout << "Frames processed: " << frames_processed << std::endl;
        std::cout << "Average time per frame: " << processing_time << " seconds" << std::endl;
        
        // Additional batch-specific metrics
        std::cout << "Thread utilization: " << total_threads_fft2 << " concurrent threads during FFT2 phase" << std::endl;
        std::cout << "Memory usage: " << (frame_size * frames_processed) / (1024*1024) << " MB for " << frames_processed << " frames" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        // Copy results back to host for validation if needed
        for (auto& frame : frames) {
            frame.copy_frame_to_host();
        }
        
        // Do not copy the frame, just use it directly for demonstration
        // This avoids potential memory issues with copying frames
        std::cout << "\nNot transferring frame back to original frame variable to avoid memory issues." << std::endl;
        
        //*********************STEP 2 PEAK DETECTION *******************
        // Batch peak detection
        std::cout << "\nExecuting Batch Peak Detection on " << frames_processed << " frames..." << std::endl;
        
        // Create vector of peakInfo structures for batch processing
        std::vector<RadarData::peakInfo> peakInfos;
        peakInfos.reserve(frames_processed);
        
        for (int i = 0; i < frames_processed; ++i) {
            peakInfos.emplace_back(rconfig.num_receivers, rconfig.num_chirps, rconfig.num_samples);
        }
        
        // Initialize persistent arrays for batch peak detection (only once)
        bool use_persistent_arrays = false;
        if (!BatchPeakDetection::persistent_arrays_initialized) {
            std::cout << "Initializing persistent arrays for batch peak detection..." << std::endl;
            if (BatchPeakDetection::initializePersistentArrays(frames, peakInfos, frames_processed)) {
                use_persistent_arrays = true;
                std::cout << "Using persistent arrays for all subsequent batch processing" << std::endl;
            }
        } else {
            use_persistent_arrays = true;
            std::cout << "Using previously initialized persistent arrays" << std::endl;
        }
        
        // Execute batch peak detection
        start = std::chrono::high_resolution_clock::now();
        BatchPeakDetection::batchPeakDetectionPipeline(frames, peakInfos, frames_processed, use_persistent_arrays);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        printTimingInfo("Batch Peak Detection (" + std::to_string(frames_processed) + " frames)", elapsed);
        double batch_peak_avg = elapsed.count() / frames_processed;
        std::cout << "Average per frame: " << batch_peak_avg << " seconds" << std::endl;
        
        // Print summary of peak detection with standardized format
        // Calculate total peaks detected in batch mode
        int total_peaks = 0;
        for (int i = 0; i < frames_processed; ++i) {
            total_peaks += peakInfos[i].num_peaks;
        }
        double avg_peaks_per_frame = static_cast<double>(total_peaks) / frames_processed;
        // Standardized output format
        std::cout << "\n======= Peak Detection Summary =======" << std::endl;
        std::cout << "Processing mode: BATCH (parallel)" << std::endl;
        std::cout << "Frames processed: " << frames_processed << std::endl;
        std::cout << "Average time per frame: " << batch_peak_avg << " seconds" << std::endl;

        // Peak information
        std::cout << "Total peaks detected: " << total_peaks << std::endl;
        std::cout << "Average peaks per frame: " << avg_peaks_per_frame << std::endl;
        std::cout << "=========================================" << std::endl;
        
        // Calculate and print overall processing time
        double total_processing_time = 0.0;
        int frames_count = 0;
        
        // Add up the times for all implemented batch processing stages
        total_processing_time += batch_avg * frames_processed;  // FFT processing time
        total_processing_time += batch_peak_avg * frames_processed;  // Peak detection time
        frames_count = frames_processed;

        // Print summary with standardized format regardless of processing mode
        std::cout << "\n======= Overall Processing Summary =======" << std::endl;
        std::cout << "Processing mode: BATCH" << std::endl;
        std::cout << "Total processing time for " << frames_count << " frames: " << std::fixed << std::setprecision(6) << total_processing_time << " seconds" << std::endl;
        std::cout << "Average time per frame: " << total_processing_time / frames_count << " seconds" << std::endl;
        std::cout << "Frames per second: " << frames_count / total_processing_time << std::endl;
        std::cout << "=============================================" << std::endl;
        
        std::cout << "\nRemaining pipeline stages are commented out." << std::endl;
        std::cout << "This implementation focuses only on batch FFT and Peak Detection processing." << std::endl;
        
        // Use the centralized batch cleanup function
        RadarData::cleanupBatchResources(
            frames,                                    // Vector of frames
            peakInfos,                                 // Vector of peak infos
            doaInfo,                                   // DoA info
            targetResults,                             // Target results
            BatchPeakDetection::persistent_arrays_initialized,  // Whether persistent arrays were initialized
            &BatchPeakDetection::cleanupPersistentArrays       // Function to cleanup persistent arrays
        );
        
        std::cout << "\nProcessing pipeline complete." << std::endl;
        
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
