
#include <iostream>
#include <chrono> // Include for timing functions
#include "config/config.hpp"
#include "data_types/datatypes.cuh"
#include "preprocessing/fft_processing.cuh"
#include "peak_detection/peak_detection.cuh"
#include "mimo_synthesis/mimo_synthesis.cuh"
#include "doa_processing/doa_processing.cuh"
#include "target_processing/target_processing.cuh"
#include "rcs/rcs.cuh"
#include "ego_estimation/ego_estimation.cuh"
#include "ghost_removal/ghost_removal.cuh"


int main() 
{
    // Load radar configuration

    RadarConfig::Config rconfig = RadarConfig::load_config();
    RadarData::Frame frame(rconfig.num_receivers, rconfig.num_chirps, rconfig.num_samples);
    std::cout << "Radar Configuration Loaded:" << std::endl;
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
        auto start = std::chrono::high_resolution_clock::now();
        fftProcessing::fftProcessPipeline(frame);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time taken for fftProcessPipeline: " << elapsed.count() << " seconds" << std::endl;
        frame.copy_frame_to_host();
    
    //*********************STEP 2 PEAK DETECTION  *******************

       start = std::chrono::high_resolution_clock::now();
       PeakDetection::cfar_peak_detection(frame, peakinfo);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        
        std::cout << "Time taken for peakDetection: " << elapsed.count() << " seconds" << std::endl;
        peakinfo.copy_peakInfo_to_host();
        std::cout << "Number of peaks detected: " << peakinfo.num_peaks << std::endl;
        // Output detected peaks
        /*for (int i = 0; i < peakinfo.num_peaks; ++i) {
            const RadarData::Peak& peak = peakinfo.peakList[i];
            std::cout << "Peak " << i + 1 << ": Receiver " << peak.receiver
                      << ", Chirp " << peak.chirp
                      << ", Sample " << peak.sample
                      << ", Value " << peak.value << std::endl;
        }*/
    
        //*********************STEP 3 MIMO SYNTHESIS PEAK SNAP DETECTION  *******************
        start = std::chrono::high_resolution_clock::now();
        MIMOSynthesis::synthesize_peaks(frame, peakinfo);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        peakinfo.copyPeakSnapsToHost();
        std::cout << "Time taken for MIMO synthesis: " << elapsed.count() << " seconds" << std::endl;
        // Output synthesized peak snaps
        for (int i = 0; i < peakinfo.num_peaks; ++i) {
            //std::cout << "Peak Snap " << i + 1 << ": ";
            for (int r = 0; r < rconfig.num_receivers; ++r) {
                //std::cout << peakinfo.peaksnaps[i * rconfig.num_receivers + r] << " ";
            }
            //std::cout << std::endl;
        }
    
    
     
        //*********************STEP 4 DOA PROCESSING  *******************
        RadarData::DoAInfo doaInfo(peakinfo.num_peaks,rconfig.num_receivers);
        doaInfo.initialize();

        start = std::chrono::high_resolution_clock::now();
        DOAProcessing::compute_music_doa(peakinfo, doaInfo,/*num_sources=*/1);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Time taken for DOA processing: " << elapsed.count() << " seconds" << std::endl;
        //*********************STEP 5 TARGET DETECTION (GPU) *******************
        RadarData::TargetResults targetResults(peakinfo.num_peaks);
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
        std::cout << "Time taken for target processing: " << elapsed.count() << " seconds" << std::endl;
        targetResults.copy_to_host();

        //*********************STEP 6 RCS ESTIMATION (GPU) *******************
        double transmittedPower = 1.0; // Example: 1 Watt
        double transmitterGain = 10.0; // Example: 10 dB
        double receiverGain = 10.0;    // Example: 10 dB
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
        targetResults.copy_to_host();
        std::cout << "Time taken for RCS estimation: " << elapsed.count() << " seconds" << std::endl;

        std::cout << "Targets detected:" << std::endl;
        for (int i = 0; i < targetResults.num_targets; ++i) {
            //const auto& target = targetResults.targets[i];
            //std::cout << "RCS: " << targetResults.targets[i].rcs << " m^2" << std::endl;
        }
       
        /*********************STEP 6 EGO ESTIMATION (GPU) *******************/
        double egoSpeed = EgoMotion::estimate_ego_motion_gpu(targetResults.d_targets, targetResults.num_targets);
        std::cout << "Estimated Ego Vehicle Speed (GPU): " << egoSpeed << " m/s" << std::endl;
        
        //*********************STEP 7 GHOST TARGET REMOVAL *******************/
        RadarData::TargetResults filteredResults = GhostRemoval::remove_ghost_targets(targetResults, egoSpeed);

        // Output filtered targets
       // std::cout << "Filtered Targets (after ghost removal):" << std::endl;
        /*for (int i = 0; i < filteredResults.num_targets; i++) {
            const auto& target = filteredResults.targets[i];
            std::cout << "Location: (" << target.x << ", " << target.y << ", " << target.z << ")"
                    << ", Range: " << target.range
                    << ", Azimuth: " << target.azimuth
                    << ", Elevation: " << target.elevation
                << ", Strength: " << target.strength
                << ", Relative Speed: " << target.relativeSpeed << std::endl;
        }*/
        std::cout << "Number of targets after ghost removal: " << filteredResults.num_targets << std::endl;
        
        //std::cout << "Processing complete. Press any key to exit..." << std::endl;
       // std::cin.get();
    }

    return 0; 
}
