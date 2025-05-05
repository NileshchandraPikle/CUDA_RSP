#ifndef MIMO_SYNTHESIS_HPP
#define MIMO_SYNTHESIS_HPP

#include "../data_types/datatypes.hpp"

namespace MIMOSynthesis {
    // Function to perform MIMO synthesis
    void synthesize_peaks(const RadarData::PeakList& peakList, const RadarData::Frame& frame, RadarData::PeakSnaps& peakSnaps);
}

#endif // MIMO_SYNTHESIS_HPP
