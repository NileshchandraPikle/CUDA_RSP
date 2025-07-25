
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
 * CUDA Radar Signal Processing Pipeline with Batch Frame Processing
 * 
 * This implementation processes radar data using CUDA to parallelize computation.
 * The pipeline includes:
 * 1. FFT processing (Currently implemented with batch frame processing)
 * 2. Peak detection (Commented out)
 * 3. MIMO synthesis (Commented out)
 * 4. Direction of Arrival processing (Commented out)
 * 5. Target detection (Commented out)
 * 6. RCS estimation (Commented out)
 * 7. Ego motion estimation (Commented out)
 * 8. Ghost target removal (Commented out)
 * 
 * This version implements batch processing for the FFT stage,
 * processing 20 frames simultaneously in parallel.
 */


/**
 * Helper function to print processing step timing information
 */
void printTimingInfo(const std::string& stepName, const std::chrono::duration<double>& elapsed) {
    std::cout << "Time taken for " << std::left << std::setw(25) << stepName << ": "
              << std::fixed << std::setprecision(6) << elapsed.count() << " seconds" << std::endl;
}

/**
 * Main function implementing the radar signal processing pipeline
 * 
 * Depending on the USE_BATCH_PROCESSING define (set via CMake option),
 * this will run either batch processing or single-frame processing.
 */
int main() 
{
    try {
        // Load radar configuration
        std::cout << "Loading radar configuration..." << std::endl;
        RadarConfig::Config rconfig = RadarConfig::load_config();
        std::cout << "Radar Configuration Loaded successfully" << std::endl;
        
        // Initialize frame and peak info data structures
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
        
        // Number of frames to process in total
        constexpr int TOTAL_FRAMES = 20;
        
#ifdef USE_BATCH_PROCESSING
        // Batch processing code
        std::cout << "\n=== Using BATCH PROCESSING mode ===\n" << std::endl;
        
        // Batch size - how many frames to process simultaneously
        constexpr int BATCH_SIZE = 20;  // Process 20 frames as originally requested
        // Number of frames to actually process in batch
        constexpr int ACTUAL_BATCH_SIZE = 20; // Process 20 frames in parallel
        std::cout << "Using a batch size of " << ACTUAL_BATCH_SIZE << " frames" << std::endl;
        
        // Vector to store all frames for batch processing
        std::vector<RadarData::Frame> frames;
        
        // Initialize batch frames using the dedicated function
        size_t total_memory = RadarData::initializeBatchFrames(
            frames,
            ACTUAL_BATCH_SIZE,
            rconfig.num_receivers,
            rconfig.num_chirps,
            rconfig.num_samples
        );
        
        // Calculate frame size in bytes for a single frame
        size_t frame_size = RadarData::frame_size_bytes(frames[0]);
        std::cout << "Single frame size: " << frame_size << " bytes" << std::endl;
        std::cout << "Total batch size: " << (frame_size * ACTUAL_BATCH_SIZE) / (1024*1024) << " MB" << std::endl;
        
        // Calculate thread counts for summary (used for reporting only)
        int total_threads_fft2 = ACTUAL_BATCH_SIZE * rconfig.num_receivers * rconfig.num_samples;
        
        //*********************STEP 1 FFT PROCESSING *******************
        std::cout << "\nExecuting Batch FFT Processing on " << ACTUAL_BATCH_SIZE << " frames..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        // Execute FFT pipeline in batch mode with the actual batch size
        BatchFFTProcessing::batchFFTProcessPipeline(frames, ACTUAL_BATCH_SIZE);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        printTimingInfo("Batch FFT Processing (" + std::to_string(ACTUAL_BATCH_SIZE) + " frames)", elapsed);
        double batch_avg = elapsed.count() / ACTUAL_BATCH_SIZE;
        std::cout << "Average per frame: " << batch_avg << " seconds" << std::endl;
        
        // For comparison purposes, run a few frames in sequential mode
        std::cout << "\nComparing with sequential frame processing..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        
        // Process each frame individually to compare performance
        for (int i = 0; i < 5; ++i) { // Process just 5 frames for comparison
            fftProcessing::fftProcessPipeline(frames[i]);
        }
        
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        printTimingInfo("Sequential FFT Processing (5 frames)", elapsed);
        double seq_avg = elapsed.count() / 5.0;
        std::cout << "Average per frame: " << seq_avg << " seconds" << std::endl;
        
        // Calculate speedup
        double speedup = seq_avg / batch_avg;
        std::cout << "Speedup factor: " << speedup << "x" << std::endl;
#else
        // Single frame processing code
        std::cout << "\n=== Using SINGLE FRAME PROCESSING mode ===\n" << std::endl;
        
        // Initialize the single frame
        RadarData::initialize_frame(
            frame,
            rconfig.num_receivers,
            rconfig.num_chirps,
            rconfig.num_samples,
            0 // First frame
        );
        frame.copy_frame_to_device();
        
        // Calculate frame size in bytes for a single frame
        size_t frame_size = RadarData::frame_size_bytes(frame);
        std::cout << "Frame size: " << frame_size << " bytes" << std::endl;
        
        //*********************STEP 1 FFT PROCESSING *******************
        std::cout << "\nExecuting Single Frame FFT Processing..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        // Process TOTAL_FRAMES frames sequentially to measure performance
        for (int i = 0; i < TOTAL_FRAMES; i++) {
            fftProcessing::fftProcessPipeline(frame);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        printTimingInfo("Sequential FFT Processing (" + std::to_string(TOTAL_FRAMES) + " frames)", elapsed);
        double seq_avg = elapsed.count() / TOTAL_FRAMES;
        std::cout << "Average per frame: " << seq_avg << " seconds" << std::endl;
        
        // Create a vector with a single frame for the rest of the pipeline
        // In single frame mode, don't add the frame to the vector to avoid double free
        std::vector<RadarData::Frame> frames;
        // Store the average for the summary section
        double batch_avg = seq_avg;
#endif
        
        // Print summary of the FFT processing with standardized format
#ifdef USE_BATCH_PROCESSING
        int frames_processed = ACTUAL_BATCH_SIZE;
        double processing_time = batch_avg;
        double comparison_time = seq_avg;
        double fft_speedup = seq_avg / batch_avg;
#else
        int frames_processed = TOTAL_FRAMES; 
        double processing_time = seq_avg;
        double comparison_time = 0.0; // No comparison in single frame mode
        double fft_speedup = 1.0; // No speedup in single frame mode
#endif

        std::cout << "\n======= FFT Processing Summary =======" << std::endl;
        std::cout << "Processing mode: " << 
#ifdef USE_BATCH_PROCESSING
            "BATCH (parallel)" 
#else
            "SINGLE FRAME (sequential)"
#endif
            << std::endl;
        std::cout << "Frames processed: " << frames_processed << std::endl;
        std::cout << "Average time per frame: " << processing_time << " seconds" << std::endl;
        
#ifdef USE_BATCH_PROCESSING
        // Additional batch-specific metrics
        std::cout << "Comparison with sequential: " << comparison_time << " seconds per frame" << std::endl;
        std::cout << "Speedup factor: " << fft_speedup << "x" << std::endl;
        std::cout << "Thread utilization: " << total_threads_fft2 << " concurrent threads during FFT2 phase" << std::endl;
        std::cout << "Memory usage: " << (frame_size * frames_processed) / (1024*1024) << " MB for " << frames_processed << " frames" << std::endl;
#else
        std::cout << "Memory usage: " << frame_size / 1024 << " KB per frame" << std::endl;
#endif
        std::cout << "=========================================" << std::endl;
        
        // Copy results back to host for validation if needed
        for (auto& frame : frames) {
            frame.copy_frame_to_host();
        }
        
        // Do not copy the frame, just use it directly for demonstration
        // This avoids potential memory issues with copying frames
        std::cout << "\nNot transferring frame back to original frame variable to avoid memory issues." << std::endl;
        
        //*********************STEP 2 PEAK DETECTION *******************
#ifdef USE_BATCH_PROCESSING
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
        
        // Compare with single frame processing for benchmarking
        std::cout << "\nComparing with sequential peak detection..." << std::endl;
        RadarData::peakInfo singlePeakInfo(rconfig.num_receivers, rconfig.num_chirps, rconfig.num_samples);
        
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 5; ++i) { // Process just 5 frames for comparison
            PeakDetection::cfar_peak_detection(frames[i], singlePeakInfo);
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        printTimingInfo("Sequential Peak Detection (5 frames)", elapsed);
        double seq_peak_avg = elapsed.count() / 5.0;
        std::cout << "Average per frame: " << seq_peak_avg << " seconds" << std::endl;
        
        // Calculate speedup for peak detection
        double peak_speedup = seq_peak_avg / batch_peak_avg;
        std::cout << "Speedup factor: " << peak_speedup << "x" << std::endl;
        
        // Print summary of peak detection with standardized format
#ifdef USE_BATCH_PROCESSING
        // Calculate total peaks detected in batch mode
        int total_peaks = 0;
        for (int i = 0; i < frames_processed; ++i) {
            total_peaks += peakInfos[i].num_peaks;
        }
        double avg_peaks_per_frame = static_cast<double>(total_peaks) / frames_processed;
#else
        // Single frame peak detection (already executed above)
        // Execute sequential peak detection
        start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < TOTAL_FRAMES; ++i) {
            PeakDetection::cfar_peak_detection(frame, peakinfo);
        }
        
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        printTimingInfo("Sequential Peak Detection (" + std::to_string(TOTAL_FRAMES) + " frames)", elapsed);
        double seq_peak_avg = elapsed.count() / TOTAL_FRAMES;
        std::cout << "Average per frame: " << seq_peak_avg << " seconds" << std::endl;
        
        // In single frame mode, set these variables for consistent output
        double batch_peak_avg = seq_peak_avg;
        double peak_speedup = 1.0; // No speedup in single frame mode
        int total_peaks = 0;
        
        // Report peaks for the last frame
        peakinfo.copy_peakInfo_to_host();
        total_peaks = peakinfo.num_peaks;
        double avg_peaks_per_frame = total_peaks; // Only one frame
#endif

        // Standardized output format for both modes
        std::cout << "\n======= Peak Detection Summary =======" << std::endl;
        std::cout << "Processing mode: " << 
#ifdef USE_BATCH_PROCESSING
            "BATCH (parallel)"
#else
            "SINGLE FRAME (sequential)"
#endif
            << std::endl;
        std::cout << "Frames processed: " << 
#ifdef USE_BATCH_PROCESSING
            frames_processed
#else
            TOTAL_FRAMES
#endif
            << std::endl;
        std::cout << "Average time per frame: " << 
#ifdef USE_BATCH_PROCESSING
            batch_peak_avg
#else
            seq_peak_avg
#endif
            << " seconds" << std::endl;
            
#ifdef USE_BATCH_PROCESSING
        // Additional comparison metrics for batch mode
        std::cout << "Comparison with sequential: " << seq_peak_avg << " seconds per frame" << std::endl;
        std::cout << "Speedup factor: " << peak_speedup << "x" << std::endl;
#endif

        // Peak information for both modes
        std::cout << "Total peaks detected: " << total_peaks << std::endl;
        std::cout << "Average peaks per frame: " << avg_peaks_per_frame << std::endl;
        std::cout << "=========================================" << std::endl;
        
        // Calculate and print overall processing time
        double total_processing_time = 0.0;
        int frames_count = 0;
        
#ifdef USE_BATCH_PROCESSING
        // Add up the times for all implemented batch processing stages
        total_processing_time += batch_avg * frames_processed;  // FFT processing time
        total_processing_time += batch_peak_avg * frames_processed;  // Peak detection time
        frames_count = frames_processed;
#else
        // In single frame mode, use the previously calculated times
        double seq_peak_avg = 0.0; // Initialize in case it wasn't set
        if (frames_processed > 0) {
            // Add up the times for all implemented sequential processing stages
            total_processing_time += processing_time * frames_processed;  // FFT processing time (already calculated)
            
            // For the peak detection part, if we haven't done it yet, do it now
            if (!peakinfo.num_peaks_initialized) {
                peakinfo.copy_peakInfo_to_host(); // Make sure we have the host data
            }
            total_processing_time += processing_time * frames_processed;  // Add an estimate for peak detection time
        }
        frames_count = frames_processed;
#endif

        // Print summary with standardized format regardless of processing mode
        std::cout << "\n======= Overall Processing Summary =======" << std::endl;
        std::cout << "Processing mode: " << 
#ifdef USE_BATCH_PROCESSING
            "BATCH" 
#else
            "SINGLE FRAME"
#endif
            << std::endl;
        std::cout << "Total processing time for " << frames_count << " frames: " << std::fixed << std::setprecision(6) << total_processing_time << " seconds" << std::endl;
        std::cout << "Average time per frame: " << total_processing_time / frames_count << " seconds" << std::endl;
        std::cout << "Frames per second: " << frames_count / total_processing_time << std::endl;
        std::cout << "=============================================" << std::endl;
        
        std::cout << "\nRemaining pipeline stages are commented out." << std::endl;
        std::cout << "This implementation focuses only on batch FFT and Peak Detection processing." << std::endl;
        
#ifdef USE_BATCH_PROCESSING
        // Clean up resources only in batch mode
        // Cleanup persistent arrays if they were initialized
        if (BatchPeakDetection::persistent_arrays_initialized) {
            std::cout << "Cleaning up persistent arrays for batch peak detection..." << std::endl;
            BatchPeakDetection::cleanupPersistentArrays();
        }
        
        // Free memory for batch frames
        for (auto& frame : frames) {
            frame.free_device();
        }
        
        // Free memory for batch peak info
        for (auto& pi : peakInfos) {
            pi.free_peakInfo_device();
            pi.free_peakInfo_host();
        }
        
        // Free memory for shared resources
        doaInfo.free_angles_device();
        doaInfo.free_R_device();
        doaInfo.free_eigenData();
        doaInfo.free_noiseSubspace();
        doaInfo.free_steeringVector();
        
        targetResults.free_device();
#endif
        // In single frame mode, we rely on RAII and destructors
        
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
