#ifndef EGO_MOTION_HPP
#define EGO_MOTION_HPP

#include "../data_types/datatypes.cuh"
#include <cuda_runtime.h>

namespace EgoMotion {
    // (Legacy CPU version removed: TargetList is obsolete)

    // GPU-parallel version
    double estimate_ego_motion_gpu(const RadarData::Target* d_targets, int num_targets, cudaStream_t stream = 0);
}

#endif // EGO_MOTION_HPP
