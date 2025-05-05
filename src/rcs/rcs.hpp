#ifndef RCS_ESTIMATION_HPP
#define RCS_ESTIMATION_HPP

#include "../target_processing/target_processing.hpp"

namespace RCSEstimation{
    // Function to estimate RCS for each target
    void estimate_rcs(TargetProcessing::TargetList& targetList,
        double transmittedPower,
        double transmitterGain,
        double receiverGain);
}
#endif // RCS_ESTIMATION_HPP

