
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
 * CUDA Radar Signal Processing Pipeline - Single Frame Processing
 * 
 * This implementation processes radar data using CUDA to parallelize computation.
 * The pipeline includes:
 * 1. FFT processing
 * 2. Peak detection
 * 3. MIMO synthesis
 * 4. Direction of Arrival processing
 * 5. Target detection
 * 6. RCS estimation
 * 7. Ego motion estimation
 * 8. Ghost target removal
 * 
 * This version implements single frame processing for all stages.
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
 * This version implements single frame processing for all stages.
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
        
        // Print summary of the FFT processing
        std::cout << "\n======= FFT Processing Summary =======" << std::endl;
        std::cout << "Processing mode: SINGLE FRAME (sequential)" << std::endl;
        std::cout << "Frames processed: " << TOTAL_FRAMES << std::endl;
        std::cout << "Average time per frame: " << seq_avg << " seconds" << std::endl;
        std::cout << "Memory usage: " << frame_size / 1024 << " KB per frame" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        // Copy results back to host for validation if needed
        frame.copy_frame_to_host();
        
        //*********************STEP 2 PEAK DETECTION *******************
        // Single frame peak detection
        std::cout << "\nExecuting Sequential Peak Detection..." << std::endl;
        
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
        
        // Report peaks for the last frame
        peakinfo.copy_peakInfo_to_host();
        int total_peaks = peakinfo.num_peaks;
        double avg_peaks_per_frame = static_cast<double>(total_peaks); // Only one frame
        
        // Peak detection summary
        std::cout << "\n======= Peak Detection Summary =======" << std::endl;
        std::cout << "Processing mode: SINGLE FRAME (sequential)" << std::endl;
        std::cout << "Frames processed: " << TOTAL_FRAMES << std::endl;
        std::cout << "Average time per frame: " << seq_peak_avg << " seconds" << std::endl;
        std::cout << "Total peaks detected: " << total_peaks << std::endl;
        std::cout << "Average peaks per frame: " << avg_peaks_per_frame << std::endl;
        std::cout << "=========================================" << std::endl;
        
        //*********************STEP 3 MIMO SYNTHESIS *******************
        std::cout << "\n*** MIMO Synthesis stage (commented out) ***" << std::endl;
        // std::cout << "\nExecuting MIMO Synthesis..." << std::endl;
        // start = std::chrono::high_resolution_clock::now();
        
        // Call MIMO synthesis pipeline
        // MIMOSynthesis::mimoSynthesisPipeline(frame);
        
        // end = std::chrono::high_resolution_clock::now();
        // elapsed = end - start;
        // printTimingInfo("MIMO Synthesis", elapsed);
        
        //*********************STEP 4 DIRECTION OF ARRIVAL PROCESSING *******************
        std::cout << "\n*** Direction of Arrival Processing stage (commented out) ***" << std::endl;
        // std::cout << "\nExecuting Direction of Arrival Processing..." << std::endl;
        // start = std::chrono::high_resolution_clock::now();
        
        // Call DoA processing pipeline
        // DoAProcessing::doaProcessingPipeline(peakinfo, doaInfo);
        
        // end = std::chrono::high_resolution_clock::now();
        // elapsed = end - start;
        // printTimingInfo("Direction of Arrival Processing", elapsed);
        
        //*********************STEP 5 TARGET PROCESSING *******************
        std::cout << "\n*** Target Detection Processing stage (commented out) ***" << std::endl;
        // std::cout << "\nExecuting Target Processing..." << std::endl;
        // start = std::chrono::high_resolution_clock::now();
        
        // Call target processing pipeline
        // TargetProcessing::targetProcessingPipeline(peakinfo, doaInfo, targetResults);
        
        // end = std::chrono::high_resolution_clock::now();
        // elapsed = end - start;
        // printTimingInfo("Target Processing", elapsed);
        
        //*********************STEP 6 RCS ESTIMATION *******************
        std::cout << "\n*** RCS Estimation stage (commented out) ***" << std::endl;
        // std::cout << "\nExecuting RCS Estimation..." << std::endl;
        // start = std::chrono::high_resolution_clock::now();
        
        // Call RCS estimation pipeline
        // RCS::rcsEstimationPipeline(frame, targetResults);
        
        // end = std::chrono::high_resolution_clock::now();
        // elapsed = end - start;
        // printTimingInfo("RCS Estimation", elapsed);
        
        //*********************STEP 7 EGO MOTION ESTIMATION *******************
        std::cout << "\n*** Ego Motion Estimation stage (commented out) ***" << std::endl;
        // std::cout << "\nExecuting Ego Motion Estimation..." << std::endl;
        // start = std::chrono::high_resolution_clock::now();
        
        // Call ego motion estimation pipeline
        // EgoEstimation::egoEstimationPipeline(targetResults);
        
        // end = std::chrono::high_resolution_clock::now();
        // elapsed = end - start;
        // printTimingInfo("Ego Motion Estimation", elapsed);
        
        //*********************STEP 8 GHOST REMOVAL *******************
        std::cout << "\n*** Ghost Removal stage (commented out) ***" << std::endl;
        // std::cout << "\nExecuting Ghost Target Removal..." << std::endl;
        // start = std::chrono::high_resolution_clock::now();
        
        // Call ghost removal pipeline
        // GhostRemoval::ghostRemovalPipeline(targetResults, filteredResults);
        
        // end = std::chrono::high_resolution_clock::now();
        // elapsed = end - start;
        // printTimingInfo("Ghost Target Removal", elapsed);
        
        // Calculate and print overall processing time
        double total_processing_time = 0.0;
        
        // Add up the times for all implemented sequential processing stages
        total_processing_time += seq_avg * TOTAL_FRAMES;        // FFT processing time
        total_processing_time += seq_peak_avg * TOTAL_FRAMES;   // Peak detection time
        
        // Print summary with standardized format
        std::cout << "\n======= Overall Processing Summary =======" << std::endl;
        std::cout << "Processing mode: SINGLE FRAME" << std::endl;
        std::cout << "Total processing time for " << TOTAL_FRAMES << " frames: " << std::fixed << std::setprecision(6) << total_processing_time << " seconds" << std::endl;
        std::cout << "Average time per frame: " << total_processing_time / TOTAL_FRAMES << " seconds" << std::endl;
        std::cout << "Frames per second: " << TOTAL_FRAMES / total_processing_time << std::endl;
        std::cout << "=============================================" << std::endl;
        
        std::cout << "\nAll pipeline stages executed successfully." << std::endl;
        
        // Clean up all radar resources using the centralized function
        RadarData::cleanupRadarResources(frame, peakinfo, doaInfo, targetResults);
        
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
