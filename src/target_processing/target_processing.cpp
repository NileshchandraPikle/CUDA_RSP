#include "target_processing.hpp"
#include "../config/config.hpp"
#include <cmath> // For trigonometric functions
#include <iostream> // For debug output
#include <complex> // Ensure this is included for std::abs with std::complex

namespace TargetProcessing {
    // Placeholder function to calculate time delay
    double calculate_time_delay(const std::vector<std::complex<double>>& snap) {
        // Example logic: Replace with actual time delay calculation
        if (snap.empty()) {
            std::cerr << "Error: Empty snapshot provided for time delay calculation." << std::endl;
            return 0.0;
        }

        // Use the magnitude of the first sample to infer time delay
        double timeDelay = std::abs(snap[0]) * 1e-9; // Example scaling factor
        return timeDelay;
    }

    // Function to calculate Doppler frequency shift
    double calculate_doppler_shift(const std::vector<std::complex<double>>& snap) {
        if (snap.empty()) {
            std::cerr << "Error: Empty snapshot provided for Doppler shift calculation." << std::endl;
            return 0.0;
        }

        // Example logic: Use phase difference between consecutive samples to estimate Doppler shift
        double dopplerShift = 0.0;
        for (size_t i = 1; i < snap.size(); ++i) {
            double phaseDiff = std::arg(snap[i]) - std::arg(snap[i - 1]);
            dopplerShift += phaseDiff;
        }
        dopplerShift /= (snap.size() - 1); // Average phase difference
        return dopplerShift;
    }

    TargetList detect_targets(const RadarData::PeakSnaps& peakSnaps,
        const std::vector<std::pair<double, double>>& doaResults) {
        TargetList targetList;

        // Radar parameters
        double wavelength = RadarConfig::WAVELENGTH;
        double d = RadarConfig::ANTENNA_SPACING;
        double c = 3e8; // Speed of light in m/s

        // Ensure the sizes of peakSnaps and doaResults match
        if (peakSnaps.size() != doaResults.size()) {
            std::cerr << "Error: Mismatch between PeakSnaps and DOA results sizes." << std::endl;
            return targetList;
        }

        // Iterate over each detected peak
        for (size_t i = 0; i < peakSnaps.size(); ++i) {
            const auto& snap = peakSnaps[i];
            const auto& doa = doaResults[i];

            // Extract azimuth and elevation
            double azimuth = doa.first;   // In degrees
            double elevation = doa.second; // In degrees

            // Calculate range using time delay
            double timeDelay = calculate_time_delay(snap);
            double range = (c * timeDelay) / 2.0;

            // Convert azimuth and elevation to radians
            double azimuthRad = azimuth * RadarConfig::PI / 180.0;
            double elevationRad = elevation * RadarConfig::PI / 180.0;

            // Convert to Cartesian coordinates
            double x = range * cos(elevationRad) * cos(azimuthRad);
            double y = range * cos(elevationRad) * sin(azimuthRad);
            double z = range * sin(elevationRad);

            // Calculate signal strength
            double strength = 0.0;
            for (const auto& value : snap) {
                strength += std::abs(value); // Explicitly use std::abs for std::complex
            }

            // Calculate relative speed using Doppler shift
            double dopplerShift = calculate_doppler_shift(snap);
            double relativeSpeed = (dopplerShift * wavelength) / 2.0;

            // Add the target to the list
            targetList.push_back({ x, y, z, range, azimuth, elevation, strength, 0.0, relativeSpeed });
        }

        return targetList;
    }
}
