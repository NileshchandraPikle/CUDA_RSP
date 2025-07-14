#ifndef DOA_PROCESSING_HPP
#define DOA_PROCESSING_HPP

#include <vector>
#include <complex>
#include <utility> // For std::pair
#include "../data_types/datatypes.cuh" // For RadarData::PeakSnaps

namespace DOAProcessing {
    // Function to compute MUSIC-based DOA
    void compute_music_doa(const RadarData::peakInfo& peakinfo,
        RadarData::DoAInfo& doAInfo, int num_sources);

    // Helper function to compute the Hermitian (conjugate transpose) of a matrix
    std::vector<std::vector<std::complex<double>>> hermitian(const std::vector<std::vector<std::complex<double>>>& matrix);

    // Helper function to multiply two matrices
    std::vector<std::vector<std::complex<double>>> multiply(const std::vector<std::vector<std::complex<double>>>& A,
        const std::vector<std::vector<std::complex<double>>>& B);

    // Helper function to compute the covariance matrix
    //std::vector<std::vector<std::complex<double>>> compute_covariance(const std::vector<std::complex<double>>& snap);
   __device__ void compute_covariance(cuDoubleComplex *snap, int num_receivers, cuDoubleComplex *d_R, int peak_index);
    // Helper function to perform eigenvalue decomposition manually
    std::pair<std::vector<double>, std::vector<std::vector<std::complex<double>>>> eigen_decomposition(
        std::vector<std::vector<std::complex<double>>>& matrix, int max_iters = 1000, double tol = 1e-6);
__global__ void compute_music_doa_kernel(
    cuDoubleComplex* d_peaksnaps, int num_peaks, cuDoubleComplex *d_R, int num_receivers, int num_sources,
    double *d_eigenvalues, cuDoubleComplex *d_eigenvector, cuDoubleComplex *d_eigenvectors, cuDoubleComplex * d_next_eigenvector,
    cuDoubleComplex *d_noiseSubspace, RadarData::DoAangles* d_angles, int max_iters = 1000, double tol = 1e-6);

    // Device function to compute covariance matrix
    __device__ void compute_covariance(cuDoubleComplex *snap, int num_receivers, cuDoubleComplex *d_R);

    // Device function to perform eigenvalue decomposition
__device__ void eigen_decomposition(cuDoubleComplex *d_R, double *d_eigenvalues, cuDoubleComplex* d_eigenvector, cuDoubleComplex *d_eigenvectors, cuDoubleComplex * d_next_eigenvector, int num_receivers, int num_peaks, int peak_index, int max_iters, double tol);
__device__ cuDoubleComplex cuCdotc(const cuDoubleComplex* a, const cuDoubleComplex* b, int n);
    }

#endif // DOA_PROCESSING_HPP
