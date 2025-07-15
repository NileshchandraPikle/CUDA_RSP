#ifndef RCS_ESTIMATION_CUH
#define RCS_ESTIMATION_CUH

#include "../data_types/datatypes.cuh"
#include "../cuda_utils/cuda_utils.hpp"

/**
 * @file rcs.cuh
 * @brief Radar Cross Section (RCS) estimation for detected targets
 */

namespace RCSEstimation {
    /**
     * @brief CUDA kernel to estimate Radar Cross Section (RCS) for targets
     *
     * This kernel calculates the radar cross section of each target based on
     * the radar equation, which relates received power, range, and system parameters.
     *
     * @param d_targets Array of targets in device memory
     * @param num_targets Number of targets to process
     * @param transmittedPower Power transmitted by the radar (in Watts)
     * @param transmitterGain Gain of the transmitter antenna
     * @param receiverGain Gain of the receiver antenna
     * @param wavelength Radar signal wavelength (in meters)
     */
    __global__ void estimate_rcs_kernel(
        RadarData::Target* d_targets,
        int num_targets,
        double transmittedPower,
        double transmitterGain,
        double receiverGain,
        double wavelength);

    /**
     * @brief Host function to estimate Radar Cross Section (RCS) for targets
     *
     * This function launches the CUDA kernel to calculate RCS for all targets.
     * The RCS is calculated using the radar equation: 
     * RCS = (received_power * (4π)³ * range⁴) / (transmitted_power * tx_gain * rx_gain * λ²)
     *
     * @param targetResults Target results structure containing target data
     * @param transmittedPower Power transmitted by the radar (in Watts)
     * @param transmitterGain Gain of the transmitter antenna
     * @param receiverGain Gain of the receiver antenna
     * @param wavelength Radar signal wavelength (in meters)
     * @throws std::runtime_error If CUDA operations fail
     */
    void estimate_rcs_gpu(
        RadarData::TargetResults& targetResults,
        double transmittedPower,
        double transmitterGain,
        double receiverGain,
        double wavelength);
}
#endif // RCS_ESTIMATION_CUH

