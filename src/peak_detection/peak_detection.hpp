#ifndef PEAK_DETECTION_HPP
#define PEAK_DETECTION_HPP

#include "../data_types/datatypes.hpp"
namespace PeakDetection {
    // Function to perform CFAR-like peak detection
    void cfar_peak_detection(const RadarData::Frame& frame, RadarData::NCI& nci, RadarData::FoldedNCI& foldedNci,
        RadarData::NoiseEstimation& noiseEstimation, RadarData::ThresholdingMap& thresholdingMap,
        RadarData::PeakList& peakList);
}

#endif // PEAK_DETECTION_HPP



