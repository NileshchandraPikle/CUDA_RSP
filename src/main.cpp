#include <iostream>
#include <chrono> // Include for timing functions
#include "config/config.hpp"
#include "data_types/datatypes.hpp"
#include "preprocessing/fft_processing.hpp"
#include "peak_detection/peak_detection.hpp"
#include "mimo_synthesis/mimo_synthesis.hpp"
#include "doa_processing/doa_processing.hpp"
#include "target_processing/target_processing.hpp" 
#include "rcs/rcs.hpp"
#include "ego_estimation/ego_estimation.hpp"
#include "ghost_removal/ghost_removal.hpp"


int main() 
{
    // Load radar configuration

    RadarConfig::Config rconfig = RadarConfig::load_config();

    // Number of frames to process
    constexpr int NUM_FRAMES = 2;
    
    // Loop over each frame
    for (int frameIndex = 0; frameIndex < NUM_FRAMES; ++frameIndex) {
        std::cout << "Processing frame " << frameIndex + 1 << " of " << NUM_FRAMES << std::endl;

        // Initialize frame by reading data for the current frame
        RadarData::Frame frame = RadarData::initialize_frame(
            rconfig.num_receivers,
            rconfig.num_chirps,
            rconfig.num_samples,
            frameIndex // Add this argument
        );

        std::cout << "Data Initialized" << std::endl;
        // Calculate frame size in bytes
        size_t frame_size = RadarData::frame_size_bytes(frame);
        std::cout << "Frame size in bytes: " << frame_size << std::endl;
    
     
        //*********************STEP 1 FFT PROCESSING *******************
        auto start = std::chrono::high_resolution_clock::now();
        fftProcessing::fftProcessPipeline(frame);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time taken for fftProcessPipeline: " << elapsed.count() << " seconds" << std::endl;
    
        
        //*********************STEP 2 PEAK DETECTION  *******************
        // Declare necessary variables for peak detection
        RadarData::NCI nci;
        RadarData::FoldedNCI foldedNci;
        RadarData::NoiseEstimation noiseEstimation;
        RadarData::ThresholdingMap thresholdingMap;
        RadarData::PeakList peakList;

        start = std::chrono::high_resolution_clock::now();
        PeakDetection::cfar_peak_detection(frame, nci, foldedNci, noiseEstimation, thresholdingMap, peakList);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Number of peaks detected: " << peakList.size() << std::endl;
        std::cout << "Time taken for peakDetection: " << elapsed.count() << " seconds" << std::endl;
    
    
        //*********************STEP 3 MIMO SYNTHESIS PEAK SNAP DETECTION  *******************
        RadarData::PeakSnaps peakSnaps;

        start = std::chrono::high_resolution_clock::now();
        MIMOSynthesis::synthesize_peaks(peakList, frame, peakSnaps);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Time taken for MIMO synthesis: " << elapsed.count() << " seconds" << std::endl;
    
     
        //*********************STEP 4 DOA PROCESSING  *******************
        std::vector<std::pair<double, double>> doaResults;

        start = std::chrono::high_resolution_clock::now();
        DOAProcessing::compute_music_doa(peakSnaps, doaResults, /*num_sources=*/1);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Time taken for DOA processing: " << elapsed.count() << " seconds" << std::endl;

        // Output DOA results for the current frame
        std::cout << "DOA Results (Azimuth, Elevation) for frame " << frameIndex + 1 << ":" <<doaResults.size()<<std::endl;
        for (const auto& result : doaResults) {
            //std::cout << "(" << result.first << ", " << result.second << ")" << std::endl;
        }
        std::cout << "peaksnap size = " << peakSnaps.size() << std::endl;
    
       
        //*********************STEP 5 TARGET DETECTION *******************
        TargetProcessing::TargetList targetList = TargetProcessing::detect_targets(peakSnaps, doaResults);

        std::cout << "Targets detected:" << std::endl;
        for (const auto& target : targetList) {
            std::cout << "Location: (" << target.x << ", " << target.y << ", " << target.z << ")"
                << ", Range: " << target.range
                << ", Azimuth: " << target.azimuth
                << ", Elevation: " << target.elevation
                << ", Strength: " << target.strength << ", Relative Speed: " << target.relativeSpeed << std::endl;
        }
    
    
    /*********************STEP 6 RADAR CROSS SECTION *******************/
     // Example radar parameters
    double transmittedPower = 1.0; // Example: 1 Watt
    double transmitterGain = 10.0; // Example: 10 dB
    double receiverGain = 10.0;    // Example: 10 dB

    // Detect targets
    TargetProcessing::TargetList targets = TargetProcessing::detect_targets(peakSnaps, doaResults);

    // Estimate RCS for each target
    RCSEstimation::estimate_rcs(targets, transmittedPower, transmitterGain, receiverGain);

    // Output results
    for (const auto& target : targets) {
        //std::cout << "Target RCS: " << target.rcs << " m^2" << std::endl;
    }
    /*********************STEP 6 EGO ESTIMATION *******************/
    double egoSpeed = EgoMotion::estimate_ego_motion(targetList);
    std::cout << "Estimated Ego Vehicle Speed: " << egoSpeed << " m/s" << std::endl;
  
  	//*********************STEP 7 GHOST TARGET REMOVAL *******************/
    TargetProcessing::TargetList filteredTargets = GhostRemoval::remove_ghost_targets(targetList, egoSpeed);

    // Output filtered targets
    std::cout << "Filtered Targets (after ghost removal):" << std::endl;
    for (const auto& target : filteredTargets) {
        std::cout << "Location: (" << target.x << ", " << target.y << ", " << target.z << ")"
            << ", Range: " << target.range
            << ", Azimuth: " << target.azimuth
            << ", Elevation: " << target.elevation
            << ", Strength: " << target.strength
            << ", Relative Speed: " << target.relativeSpeed << std::endl;
    }
	std::cout << "Number of targets after ghost removal: " << filteredTargets.size() << std::endl;
  }
    // Keep the terminal display until a key is pressed
    std::cout << "Processing complete. Press any key to exit..." << std::endl;
    std::cin.get();

    return 0;
}
