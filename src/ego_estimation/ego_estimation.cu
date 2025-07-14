#include "ego_estimation.cuh"
#include <cmath> // For std::abs
#include <cuda_runtime.h>

namespace EgoMotion {


    // GPU kernel for reduction
    __global__ void sum_valid_relative_speeds(const RadarData::Target* targets, int num_targets, double* sum, int* count) {
        __shared__ double local_sum[256];
        __shared__ int local_count[256];
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        local_sum[tid] = 0.0;
        local_count[tid] = 0;
        if (idx < num_targets) {
            double rel = targets[idx].relativeSpeed;
            if (fabs(rel) > 0.1) {
                local_sum[tid] = rel;
                local_count[tid] = 1;
            }
        }
        __syncthreads();
        // Reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                local_sum[tid] += local_sum[tid + stride];
                local_count[tid] += local_count[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            atomicAdd(sum, local_sum[0]);
            atomicAdd(count, local_count[0]);
        }
    }

    double* allocate_device_sum() {
        double* d_sum = nullptr;
        cudaMalloc(&d_sum, sizeof(double));
        return d_sum;
    }

    int* allocate_device_count() {
        int* d_count = nullptr;
        cudaMalloc(&d_count, sizeof(int));
        return d_count;
    }

    void zero_device_sum_and_count(double* d_sum, int* d_count, cudaStream_t stream = 0) {
        cudaMemsetAsync(d_sum, 0, sizeof(double), stream);
        cudaMemsetAsync(d_count, 0, sizeof(int), stream);
    }

    void launch_ego_motion_kernel(const RadarData::Target* d_targets, int num_targets, double* d_sum, int* d_count, cudaStream_t stream = 0) {
        int blockSize = 256;
        int gridSize = (num_targets + blockSize - 1) / blockSize;
        sum_valid_relative_speeds<<<gridSize, blockSize, 0, stream>>>(d_targets, num_targets, d_sum, d_count);
    }

    void copy_sum_and_count_to_host(const double* d_sum, const int* d_count, double& h_sum, int& h_count, cudaStream_t stream = 0) {
        cudaMemcpyAsync(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    void free_device_sum_and_count(double* d_sum, int* d_count) {
        cudaFree(d_sum);
        cudaFree(d_count);
    }

    double estimate_ego_motion_gpu(const RadarData::Target* d_targets, int num_targets, cudaStream_t stream) {
        double* d_sum = allocate_device_sum();
        int* d_count = allocate_device_count();
        zero_device_sum_and_count(d_sum, d_count, stream);
        launch_ego_motion_kernel(d_targets, num_targets, d_sum, d_count, stream);
        double h_sum = 0.0;
        int h_count = 0;
        copy_sum_and_count_to_host(d_sum, d_count, h_sum, h_count, stream);
        free_device_sum_and_count(d_sum, d_count);
        if (h_count == 0) return 0.0;
        return h_sum / h_count;
    }

} // namespace EgoMotion
