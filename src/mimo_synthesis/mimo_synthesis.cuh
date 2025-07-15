#ifndef MIMO_SYNTHESIS_HPP
#define MIMO_SYNTHESIS_HPP

#include "../data_types/datatypes.cuh"
#include "../cuda_utils/cuda_utils.hpp"

namespace MIMOSynthesis {
    /**
     * @brief Perform MIMO synthesis on radar peaks
     *
     * This function extracts signal data at peak locations from all receivers
     * to create virtual array snapshots for direction finding.
     *
     * @param frame Input radar frame containing signal data
     * @param peakinfo Peak information structure for input peaks and output synthesized data
     * @throws std::runtime_error If CUDA operations fail
     */
    void synthesize_peaks(const RadarData::Frame& frame, RadarData::peakInfo& peakinfo);

    /**
     * @brief CUDA kernel to perform MIMO synthesis
     *
     * For each peak, this kernel extracts the signal value at the peak location
     * from all receivers to create a virtual array snapshot.
     *
     * @param d_data Input radar data in device memory
     * @param d_peakList List of peaks in device memory
     * @param d_peaksnaps Output array for synthesized peaks in device memory
     * @param num_peaks Number of peaks to process
     * @param num_receivers Number of receivers in the array
     * @param num_chirps Number of chirps in the frame
     * @param num_samples Number of samples per chirp
     * @param max_num_peaks Maximum number of peaks (buffer size)
     */
    __global__ void synthesize_peaks_kernel(
        const cuDoubleComplex* d_data,
        RadarData::Peak* d_peakList,
        cuDoubleComplex* d_peaksnaps,
        int num_peaks,
        int num_receivers,
        int num_chirps,
        int num_samples,
        int max_num_peaks);
}
#endif // MIMO_SYNTHESIS_HPP
