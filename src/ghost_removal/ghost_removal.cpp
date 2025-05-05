#include "ghost_removal.hpp"
#include <cmath> // For std::abs

namespace GhostRemoval {

    TargetProcessing::TargetList remove_ghost_targets(
        const TargetProcessing::TargetList& targets,
        double egoSpeed) {

        TargetProcessing::TargetList filteredTargets;

        // Threshold for relative speed to identify ghost targets
        constexpr double RELATIVE_SPEED_THRESHOLD = 5.0; // Example: 5 m/s

        for (const auto& target : targets) {
            // Calculate the absolute difference between target's relative speed and ego speed
            double relativeSpeedDifference = std::abs(target.relativeSpeed - egoSpeed);

            // If the relative speed difference is below the threshold, keep the target
            if (relativeSpeedDifference > RELATIVE_SPEED_THRESHOLD) {
                // Likely a ghost target, skip it
                continue;
            }

            // Add valid target to the filtered list
            filteredTargets.push_back(target);
        }

        return filteredTargets;
    }

} // namespace GhostRemoval
