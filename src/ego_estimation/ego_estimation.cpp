#include "ego_estimation.hpp"
#include <cmath> // For std::abs

namespace EgoMotion {

    double estimate_ego_motion(const TargetProcessing::TargetList& targets) {
        if (targets.empty()) {
            return 0.0; // No targets, assume no motion
        }

        double cumulativeRelativeSpeed = 0.0;
        int validTargetCount = 0;

        for (const auto& target : targets) {
            // Use the relative velocity of the target to estimate ego motion
            // Assuming target.relativeSpeed is available in the Target structure
            if (std::abs(target.relativeSpeed) > 0.1) { // Ignore near-zero relative speeds
                cumulativeRelativeSpeed += target.relativeSpeed;
                ++validTargetCount;
            }
        }

        if (validTargetCount == 0) {
            return 0.0; // No valid targets for motion estimation
        }

        // Estimate ego speed as the average of relative speeds
        return cumulativeRelativeSpeed / validTargetCount;
    }

} // namespace EgoMotion
