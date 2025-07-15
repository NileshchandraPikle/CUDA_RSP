
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <iomanip>

// Configuration includes
#include "config/config.hpp"

// Data types and utilities
#include "data_types/datatypes.cuh"

// Processing steps includes
#include "preprocessing/fft_processing.cuh"
#include "peak_detection/peak_detection.cuh"
#include "mimo_synthesis/mimo_synthesis.cuh"
#include "doa_processing/doa_processing.cuh"
#include "target_processing/target_processing.cuh"
#include "rcs/rcs.cuh"
#include "ego_estimation/ego_estimation.cuh"
#include "ghost_removal/ghost_removal.cuh"

/**
 * CUDA Radar Signal Processing Pipeline
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

     // Number of frames to process
    constexpr int NUM_FRAMES = 2;
    for (int frameIndex = 0; frameIndex < NUM_FRAMES; ++frameIndex) {
        std::cout << "Processing frame " << frameIndex + 1 << " of " << NUM_FRAMES << std::endl;

        // Initialize frame by reading data for the current frame
        RadarData::initialize_frame(
            frame,
            rconfig.num_receivers,
            rconfig.num_chirps,
            rconfig.num_samples,
            frameIndex
        );

        //std::cout << "Data Initialized" << std::endl;
        // Calculate frame size in bytes
        size_t frame_size = RadarData::frame_size_bytes(frame);
        //std::cout << "Frame size in bytes: " << frame_size << std::endl;
        frame.copy_frame_to_device();
      
       
        //*********************STEP 1 FFT PROCESSING *******************
        std::cout << "\nExecuting FFT Processing..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        // Execute FFT pipeline (includes Hilbert transform, FFT1, FFT2)
        fftProcessing::fftProcessPipeline(frame);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        printTimingInfo("FFT Processing", elapsed);
        
        // Copy results back to host for validation if needed
        frame.copy_frame_to_host();
    
    //*********************STEP 2 PEAK DETECTION  *******************
        std::cout << "\nExecuting Peak Detection using CFAR..." << std::endl;
        
        start = std::chrono::high_resolution_clock::now();
        // Perform CFAR peak detection on GPU
        PeakDetection::cfar_peak_detection(frame, peakinfo);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        
        // Copy results back to host for validation
        peakinfo.copy_peakInfo_to_host();
        
        printTimingInfo("Peak Detection", elapsed);
        std::cout << "Number of peaks detected: " << peakinfo.num_peaks << std::endl;
        
        // Validate peak count is reasonable
        if (peakinfo.num_peaks == 0) {
            std::cerr << "Warning: No peaks detected. Check CFAR parameters." << std::endl;
        } else if (peakinfo.num_peaks >= peakinfo.max_num_peaks) {
            std::cerr << "Warning: Maximum peak limit reached. Some peaks may be truncated." << std::endl;
        }
        // Output detected peaks
        /*for (int i = 0; i < peakinfo.num_peaks; ++i) {
            const RadarData::Peak& peak = peakinfo.peakList[i];
            std::cout << "Peak " << i + 1 << ": Receiver " << peak.receiver
                      << ", Chirp " << peak.chirp
                      << ", Sample " << peak.sample
                      << ", Value " << peak.value << std::endl;
        }*/
    
        //*********************STEP 3 MIMO SYNTHESIS PEAK SNAP DETECTION  *******************
        std::cout << "\nExecuting MIMO Synthesis..." << std::endl;
        
        start = std::chrono::high_resolution_clock::now();
        // Create peak snapshots from detected peaks
        MIMOSynthesis::synthesize_peaks(frame, peakinfo);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        
        // Copy results back to host for validation
        peakinfo.copyPeakSnapsToHost();
        
        printTimingInfo("MIMO Synthesis", elapsed);
        
        // Validate peak snaps (uncomment for debugging)
        /*
        for (int i = 0; i < std::min(peakinfo.num_peaks, 5); ++i) {  // Show first 5 peaks
            std::cout << "Peak Snap " << i + 1 << " sample values: ";
            for (int r = 0; r < std::min(rconfig.num_receivers, 3); ++r) {  // Show first 3 receivers
                std::complex<double> val = peakinfo.peaksnaps[i * rconfig.num_receivers + r];
                std::cout << "(" << val.real() << ", " << val.imag() << ") ";
            }
            if (rconfig.num_receivers > 3) std::cout << "...";  // Indicate truncation
            std::cout << std::endl;
        }
        if (peakinfo.num_peaks > 5) std::cout << "... (remaining peaks truncated)" << std::endl;
        */
    
    
     
        //*********************STEP 4 DOA PROCESSING  *******************
        std::cout << "\nExecuting Direction of Arrival (DoA) Processing..." << std::endl;
        
        // Initialize DoA data structures
        RadarData::DoAInfo doaInfo(peakinfo.num_peaks, rconfig.num_receivers);
        doaInfo.initialize();

        // Process DoA using MUSIC algorithm
        start = std::chrono::high_resolution_clock::now();
        
        // Number of signal sources = 1 (assumes one target per peak)
        const int numSources = 1;
        DOAProcessing::compute_music_doa(peakinfo, doaInfo, numSources);
        
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        
        printTimingInfo("DoA Processing", elapsed);
        
        // Display DoA results (uncomment for debugging)
        /*
        std::cout << "DoA Results (Azimuth, Elevation):" << std::endl;
        for (int i = 0; i < std::min(peakinfo.num_peaks, 10); ++i) {
            std::cout << "Peak " << i + 1 << ": Azimuth = " << doaInfo.angles[i].azimuth 
                      << ", Elevation = " << doaInfo.angles[i].elevation << std::endl;
        }
        if (peakinfo.num_peaks > 10) std::cout << "... (remaining DoA results truncated)" << std::endl;
        */
        //*********************STEP 5 TARGET DETECTION (GPU) *******************
        std::cout << "\nExecuting Target Detection..." << std::endl;
        
        // Initialize target results storage
        RadarData::TargetResults targetResults(peakinfo.num_peaks);
        
        // Perform target detection on GPU
        start = std::chrono::high_resolution_clock::now();
        TargetProcessing::detect_targets_gpu(
            peakinfo.d_peaksnaps,
            doaInfo.d_angles,
            peakinfo.num_peaks,
            rconfig.num_receivers,
            targetResults
        );
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        
        // Copy results back to host
        targetResults.copy_to_host();
        
        printTimingInfo("Target Detection", elapsed);
        
        // Display target information (uncomment for debugging)
        /*
        std::cout << "Targets Detected:" << std::endl;
        for (int i = 0; i < std::min(targetResults.num_targets, 10); ++i) {
            const auto& target = targetResults.targets[i];
            std::cout << "Target " << i + 1 << ": Position (" 
                      << target.x << ", " << target.y << ", " << target.z
                      << "), Range: " << target.range 
                      << ", Speed: " << target.relativeSpeed << std::endl;
        }
        if (targetResults.num_targets > 10) std::cout << "... (remaining targets truncated)" << std::endl;
        */

        //*********************STEP 6 RCS ESTIMATION (GPU) *******************
        std::cout << "\nExecuting Radar Cross Section (RCS) Estimation..." << std::endl;
        
        // Set radar parameters for RCS estimation
        const double transmittedPower = 1.0; // Transmitted power in Watts
        const double transmitterGain = 10.0; // Transmitter gain in dB
        const double receiverGain = 10.0;    // Receiver gain in dB
        
        // Perform RCS estimation on GPU
        start = std::chrono::high_resolution_clock::now();
        RCSEstimation::estimate_rcs_gpu(
            targetResults,
            transmittedPower,
            transmitterGain,
            receiverGain,
            RadarConfig::WAVELENGTH
        );
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        
        // Copy results back to host
        targetResults.copy_to_host();
        
        printTimingInfo("RCS Estimation", elapsed);
        std::cout << "Targets detected: " << targetResults.num_targets << std::endl;
        
        // Display RCS results (uncomment for debugging)
        /*
        std::cout << "RCS Estimation Results:" << std::endl;
        for (int i = 0; i < std::min(targetResults.num_targets, 10); ++i) {
            const auto& target = targetResults.targets[i];
            std::cout << "Target " << i + 1 << ": RCS = " << target.rcs << " m²" << std::endl;
        }
        if (targetResults.num_targets > 10) std::cout << "... (remaining RCS results truncated)" << std::endl;
        */
       
        /*********************STEP 7 EGO ESTIMATION (GPU) *******************/
        std::cout << "\nExecuting Ego Motion Estimation..." << std::endl;
        
        // Estimate ego vehicle speed using GPU
        start = std::chrono::high_resolution_clock::now();
        double egoSpeed = EgoMotion::estimate_ego_motion_gpu(targetResults.d_targets, targetResults.num_targets);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        
        printTimingInfo("Ego Motion Estimation", elapsed);
        std::cout << "Estimated Ego Vehicle Speed: " << std::fixed << std::setprecision(2) 
                  << egoSpeed << " m/s" << std::endl;
        
        //*********************STEP 8 GHOST TARGET REMOVAL *******************/
        std::cout << "\nExecuting Ghost Target Removal..." << std::endl;
        
        // Create a pointer to store filtered results - helps avoid double-free issues
        RadarData::TargetResults* pFilteredResults = nullptr;
        
        start = std::chrono::high_resolution_clock::now();
        
        // Create filtered results on the heap
        pFilteredResults = new RadarData::TargetResults(targetResults.num_targets);
        
        // Only use host memory since we'll just be reading, not doing CUDA operations
        pFilteredResults->free_device();
        pFilteredResults->num_targets = 0;
        
        // Manually filter targets
        int removed = 0;
        constexpr double RELATIVE_SPEED_THRESHOLD = 5.0; // Match the threshold in ghost_removal.cu
        
        std::cout << "Ghost removal: Processing " << targetResults.num_targets << " targets" << std::endl;
        
        for (int i = 0; i < targetResults.num_targets; i++) {
            const auto& target = targetResults.targets[i];
            double relativeSpeedDifference = std::abs(target.relativeSpeed - egoSpeed);
            
            // Filter out ghost targets
            if (relativeSpeedDifference > RELATIVE_SPEED_THRESHOLD) {
                removed++;
                continue;
            }
            
            // Keep valid targets
            pFilteredResults->targets[pFilteredResults->num_targets++] = target;
        }
        
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        
        printTimingInfo("Ghost Target Removal", elapsed);
        std::cout << "Ghost removal: Removed " << removed << " targets, kept " 
                  << pFilteredResults->num_targets << " targets" << std::endl;
                  
        std::cout << "Initial targets: " << targetResults.num_targets << std::endl;
        std::cout << "Targets after ghost removal: " << pFilteredResults->num_targets << std::endl;
        std::cout << "Ghost targets removed: " << (targetResults.num_targets - pFilteredResults->num_targets) << std::endl;
        
        // Display filtered target results (uncomment for debugging)
        /*
        std::cout << "\nFiltered Targets (after ghost removal):" << std::endl;
        for (int i = 0; i < std::min(filteredResults.num_targets, 10); i++) {
            const auto& target = filteredResults.targets[i];
            std::cout << "Target " << i + 1 << ": Location (" 
                      << target.x << ", " << target.y << ", " << target.z << ")"
                      << ", Range: " << target.range
                      << ", Speed: " << target.relativeSpeed << std::endl;
        }
        if (filteredResults.num_targets > 10) std::cout << "... (remaining targets truncated)" << std::endl;
        */
        
        std::cout << "\nRadar processing pipeline complete." << std::endl;
        
        // Clean up filtered results memory
        if (pFilteredResults) {
            delete pFilteredResults;
            pFilteredResults = nullptr;
        }
    } // End frame processing loop
    
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
