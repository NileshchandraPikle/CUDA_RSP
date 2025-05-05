#ifndef EGO_MOTION_HPP
#define EGO_MOTION_HPP

#include "../target_processing/target_processing.hpp"

namespace EgoMotion {
    // Function to estimate ego vehicle speed
    double estimate_ego_motion(const TargetProcessing::TargetList& targets);
}

#endif // EGO_MOTION_HPP
