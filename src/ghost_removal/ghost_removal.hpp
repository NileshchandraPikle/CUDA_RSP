#ifndef GHOST_REMOVAL_HPP
#define GHOST_REMOVAL_HPP

#include "../target_processing/target_processing.hpp"

namespace GhostRemoval {
    // Function to remove ghost targets
    TargetProcessing::TargetList remove_ghost_targets(
        const TargetProcessing::TargetList& targets,
        double egoSpeed);
}

#endif // GHOST_REMOVAL_HPP
