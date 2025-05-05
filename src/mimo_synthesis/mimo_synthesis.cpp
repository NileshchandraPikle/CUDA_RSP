#include <iostream>
#include <vector>
#include <complex>
#include "../config/config.hpp"
#include "mimo_synthesis.hpp"

namespace MIMOSynthesis {
    void synthesize_peaks(const RadarData::PeakList& peakList, const RadarData::Frame& frame, RadarData::PeakSnaps& peakSnaps) {
        // Clear the output PeakSnaps
        peakSnaps.clear();

        // Iterate over the Peak List
        for (const auto& peak : peakList) {
            int receiver = std::get<0>(peak);
            int chirp = std::get<1>(peak);
            int sample = std::get<2>(peak);

            // Validate indices
            if (receiver < 0 || receiver >= frame.size() ||
                chirp < 0 || chirp >= frame[0].size() ||
                sample < 0 || sample >= frame[0][0].size()) {
                std::cerr << "Invalid peak indices: (" << receiver << ", " << chirp << ", " << sample << ")" << std::endl;
                continue;
            }

            // Combine data across all receivers for the given chirp and sample
            std::vector<std::complex<double>> combinedData;
            for (int r = 0; r < frame.size(); ++r) {
                combinedData.push_back(frame[r][chirp][sample]);
            }

            // Store the combined data as a Peak Snap
            peakSnaps.push_back(combinedData);
        }
    }
}
