#ifndef RCS_ESTIMATION_CUH
#define RCS_ESTIMATION_CUH

#include "../data_types/datatypes.cuh"

namespace RCSEstimation {
    __global__ void estimate_rcs_kernel(
        RadarData::Target* d_targets,
        int num_targets,
        double transmittedPower,
        double transmitterGain,
        double receiverGain,
        double wavelength);

    void estimate_rcs_gpu(
        RadarData::TargetResults& targetResults,
        double transmittedPower,
        double transmitterGain,
        double receiverGain,
        double wavelength);
}
#endif // RCS_ESTIMATION_CUH

