#ifndef PEAK_DETECTION_HPP
#define PEAK_DETECTION_HPP

#include "../data_types/datatypes.cuh"
namespace PeakDetection {
    // Function to perform CFAR-like peak detection
    void cfar_peak_detection(const RadarData::Frame& frame, RadarData::peakInfo& peakinfo);
    __global__ void cfar_peak_detection_Kernel(double* d_nci, double* d_foldedNci, double* d_noiseEstimation, double* d_thresholdingMap, RadarData::Peak* d_peakList,
        const cuDoubleComplex* d_data, int num_receivers, int num_chirps, int num_samples, double alpha, int max_num_peaks, int * d_peak_counter);
}

#endif // PEAK_DETECTION_HPP