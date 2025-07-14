
#ifndef TARGET_PROCESSING_CUH
#define TARGET_PROCESSING_CUH

#include "../data_types/datatypes.cuh"

namespace TargetProcessing {
    // CUDA kernel for target detection
    __global__ void detect_targets_kernel(
        const cuDoubleComplex* d_peaksnaps,
        const RadarData::DoAangles* d_angles,
        int num_peaks,
        int num_receivers,
        RadarData::Target* d_targets
    );

    // Host wrapper for target detection
    void detect_targets_gpu(
        cuDoubleComplex* d_peaksnaps,
        RadarData::DoAangles* d_angles,
        int num_peaks,
        int num_receivers,
        RadarData::TargetResults& targetResults
    );
}

#endif // TARGET_PROCESSING_CUH
