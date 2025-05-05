#ifndef TARGET_PROCESSING_HPP
#define TARGET_PROCESSING_HPP

#include "../data_types/datatypes.hpp"
#include <vector>
#include <utility>

namespace TargetProcessing {
    // Structure to represent a detected target
    struct Target {
        double x, y, z;          // Cartesian coordinates
        double range;            // Range in meters
        double azimuth;          // Azimuth angle in degrees
        double elevation;        // Elevation angle in degrees
        double strength;         // Signal strength
        double rcs;
        double relativeSpeed;
    };

    using TargetList = std::vector<Target>;

    // Function to detect targets from peak snapshots and DOA results
    TargetList detect_targets(const RadarData::PeakSnaps& peakSnaps,
        const std::vector<std::pair<double, double>>& doaResults);

    // Function to calculate time delay (placeholder for actual implementation)
    double calculate_time_delay(const RadarData::PeakSnap& snap);
}

#endif // TARGET_PROCESSING_HPP
