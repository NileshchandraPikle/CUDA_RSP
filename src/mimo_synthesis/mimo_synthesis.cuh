#ifndef MIMO_SYNTHESIS_HPP
#define MIMO_SYNTHESIS_HPP

#include "../data_types/datatypes.cuh"

namespace MIMOSynthesis {
    // Function to perform MIMO synthesis
    void synthesize_peaks(const RadarData::Frame& frame, RadarData::peakInfo& peakinfo);
    __global__ void synthesize_peaks_kernel(const cuDoubleComplex* d_data, RadarData::Peak* d_peakList,cuDoubleComplex* d_peaksnaps, int num_peaks, int num_receivers, int num_chirps, int num_samples, int max_num_peaks);
}
#endif // MIMO_SYNTHESIS_HPP
