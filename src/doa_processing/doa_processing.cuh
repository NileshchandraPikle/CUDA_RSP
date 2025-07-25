#ifndef DOA_PROCESSING_HPP
#define DOA_PROCESSING_HPP

#include <vector>
#include <complex>
#include <utility> // For std::pair
#include "../data_types/datatypes.cuh"
#include "../cuda_utils/cuda_utils.hpp"

/**
 * @file doa_processing.cuh
 * @brief Direction of Arrival (DoA) estimation using MUSIC algorithm
 */

namespace DOAProcessing {
    /**
     * @brief Compute Direction of Arrival using MUSIC algorithm
     *
     * This function estimates the angles of arrival for radar targets using
     * the MUSIC (MUltiple SIgnal Classification) algorithm on GPU.
     *
     * @param peakinfo Input peak information containing peak snapshots
     * @param doAInfo Output structure to store DoA results
     * @param num_sources Number of signal sources to estimate
     * @throws std::runtime_error If CUDA operations fail
     */
    void compute_music_doa(const RadarData::peakInfo& peakinfo,
                           RadarData::DoAInfo& doAInfo, 
                           int num_sources);

    /**
     * @brief Device function to compute covariance matrix
     *
     * @param snap Input signal snapshot
     * @param num_receivers Number of receivers
     * @param d_R Output covariance matrix
     * @param peak_index Index of the peak being processed
     */
    __device__ void compute_covariance(
        cuDoubleComplex *snap,
        int num_receivers,
        cuDoubleComplex *d_R,
        int peak_index);

    /**
     * @brief Helper function to perform eigenvalue decomposition manually
     *
     * @param matrix Input matrix
     * @param max_iters Maximum number of iterations
     * @param tol Convergence tolerance
     * @return Pair of eigenvalues vector and eigenvectors matrix
     */
    std::pair<std::vector<double>, std::vector<std::vector<std::complex<double>>>> eigen_decomposition(
        std::vector<std::vector<std::complex<double>>>& matrix,
        int max_iters = 1000,
        double tol = 1e-6);

    /**
     * @brief CUDA kernel to compute MUSIC-based DoA
     *
     * @param d_peaksnaps Input peak snapshots from all receivers
     * @param num_peaks Number of peaks to process
     * @param d_R Temporary covariance matrix storage
     * @param num_receivers Number of receivers
     * @param num_sources Number of signal sources to estimate
     * @param d_eigenvalues Output eigenvalues
     * @param d_eigenvector Temporary eigenvector storage
     * @param d_eigenvectors Output eigenvectors matrix
     * @param d_next_eigenvector Temporary storage for next eigenvector iteration
     * @param d_noiseSubspace Output noise subspace
     * @param d_angles Output DoA angles
     * @param max_iters Maximum number of iterations for eigendecomposition
     * @param tol Convergence tolerance
     */
    __global__ void compute_music_doa_kernel(
        cuDoubleComplex* d_peaksnaps,
        int num_peaks,
        cuDoubleComplex *d_R,
        int num_receivers,
        int num_sources,
        double *d_eigenvalues,
        cuDoubleComplex *d_eigenvector,
        cuDoubleComplex *d_eigenvectors,
        cuDoubleComplex *d_next_eigenvector,
        cuDoubleComplex *d_noiseSubspace,
        RadarData::DoAangles* d_angles,
        int max_iters = 1000,
        double tol = 1e-6);

    /**
     * @brief Device function to compute covariance matrix
     *
     * @param snap Input signal snapshot
     * @param num_receivers Number of receivers
     * @param d_R Output covariance matrix
     */
    __device__ void compute_covariance(
        cuDoubleComplex *snap,
        int num_receivers,
        cuDoubleComplex *d_R);

    /**
     * @brief Device function to perform eigenvalue decomposition
     *
     * @param d_R Input covariance matrix
     * @param d_eigenvalues Output eigenvalues
     * @param d_eigenvector Temporary eigenvector storage
     * @param d_eigenvectors Output eigenvectors matrix
     * @param d_next_eigenvector Temporary storage for next eigenvector iteration
     * @param num_receivers Number of receivers (matrix dimension)
     * @param num_peaks Number of peaks being processed
     * @param peak_index Index of current peak
     * @param max_iters Maximum number of iterations
     * @param tol Convergence tolerance
     */
    __device__ void eigen_decomposition(
        cuDoubleComplex *d_R,
        double *d_eigenvalues,
        cuDoubleComplex* d_eigenvector,
        cuDoubleComplex *d_eigenvectors,
        cuDoubleComplex *d_next_eigenvector,
        int num_receivers,
        int num_peaks,
        int peak_index,
        int max_iters,
        double tol);

    /**
     * @brief Device function to compute the conjugate dot product of two complex vectors
     *
     * @param a First vector
     * @param b Second vector
     * @param n Vector length
     * @return Complex dot product
     */
    __device__ cuDoubleComplex cuCdotc(
        const cuDoubleComplex* a,
        const cuDoubleComplex* b,
        int n);
}

#endif // DOA_PROCESSING_HPP
