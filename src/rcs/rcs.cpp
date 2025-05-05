#include "rcs.hpp"
#include "../config/config.hpp"
#include <cmath> // For mathematical operations
#include <iostream> // For debug output

namespace RCSEstimation {
    void estimate_rcs(TargetProcessing::TargetList& targetList,
        double transmittedPower,
        double transmitterGain,
        double receiverGain) {
        double wavelength = RadarConfig::WAVELENGTH;

        for (auto& target : targetList) {
            // Calculate RCS using the formula
            double receivedPower = target.strength; // Assuming strength represents received power
            double range = target.range;

            if (range <= 0.0) {
                std::cerr << "Error: Invalid range for target. Skipping RCS calculation." << std::endl;
                target.rcs = 0.0;
                continue;
            }

            target.rcs = (receivedPower * std::pow(4 * RadarConfig::PI, 3) * std::pow(range, 4)) /
                (transmittedPower * transmitterGain * receiverGain * std::pow(wavelength, 2));
        }
    }
}
