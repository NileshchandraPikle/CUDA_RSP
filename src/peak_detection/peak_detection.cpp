#include "peak_detection.hpp"
#include <cmath> // Include for std::abs
#include <tuple> // Include for std::make_tuple
#include "../config/config.hpp"
#include "../data_types/datatypes.hpp"

namespace PeakDetection {
    // Function to perform 2D CFAR-like peak detection
    void cfar_peak_detection(const RadarData::Frame& frame, RadarData::NCI& nci, RadarData::FoldedNCI& foldedNci,
        RadarData::NoiseEstimation& noiseEstimation, RadarData::ThresholdingMap& thresholdingMap,
        RadarData::PeakList& peakList) {
        int num_receivers = frame.size();
        int num_chirps = frame[0].size();
        int num_samples = frame[0][0].size();

        // Initialize the output structures
        nci.resize(num_chirps, std::vector<RadarData::Real>(num_samples, 0));
        foldedNci.resize(num_chirps, std::vector<RadarData::Real>(num_samples, 0));
        noiseEstimation.resize(num_chirps, std::vector<RadarData::Real>(num_samples, 0));
        thresholdingMap.resize(num_chirps, std::vector<RadarData::Real>(num_samples, 0));

        // CFAR parameters
        
        double alpha = RadarConfig::TRAINING_CELLS  * (std::pow(RadarConfig::FALSE_ALARM_RATE, -1.0 / RadarConfig::TRAINING_CELLS) - 1);

        // Perform CFAR detection for each receiver
        for (int r = 0; r < num_receivers; r++) {
            for (int c = 0; c < num_chirps; c++) {
                for (int s = 0; s < num_samples; s++) {
                    double magnitude = std::abs(frame[r][c][s]);

                    // Calculate noise level using training cells in both Doppler and range dimensions
                    double noise_level = 0.0;
                    int training_count = 0;

                    for (int tc = -RadarConfig::TRAINING_CELLS; tc <= RadarConfig::TRAINING_CELLS; tc++) {
                        for (int ts = -RadarConfig::TRAINING_CELLS; ts <= RadarConfig::TRAINING_CELLS; ts++) {
                            if ((tc == 0 && ts == 0) ||
                                (std::abs(tc) <= RadarConfig::GUARD_CELLS && std::abs(ts) <= RadarConfig::GUARD_CELLS)) {
                                continue; // Skip guard cells and the cell under test
                            }

                            int doppler_index = c + tc;
                            int range_index = s + ts;

                            if (doppler_index >= 0 && doppler_index < num_chirps &&
                                range_index >= 0 && range_index < num_samples) {
                                noise_level += std::abs(frame[r][doppler_index][range_index]);
                                training_count++;
                            }
                        }
                    }

                    // Calculate average noise level
                    nci[c][s] = noise_level / training_count; // Average noise level
                    foldedNci[c][s] = noise_level;
                    noise_level /= training_count;
                    noiseEstimation[c][s] = noise_level;

                    // Calculate threshold
                    double threshold = alpha * noise_level;
                    thresholdingMap[c][s] = threshold;

                    // Detect peak
                    if (magnitude > threshold) {
                        peakList.push_back(std::make_tuple(r, c, s)); // Include receiver index
                    }
                }
            }
        }
    }
}
