#ifndef DOA_PROCESSING_HPP
#define DOA_PROCESSING_HPP

#include <vector>
#include <complex>
#include <utility> // For std::pair
#include "../data_types/datatypes.hpp" // For RadarData::PeakSnaps

namespace DOAProcessing {
    // Function to compute MUSIC-based DOA
    void compute_music_doa(const RadarData::PeakSnaps& peakSnaps,
        std::vector<std::pair<double, double>>& doaResults,
        int num_sources);

    // Helper function to compute the Hermitian (conjugate transpose) of a matrix
    std::vector<std::vector<std::complex<double>>> hermitian(const std::vector<std::vector<std::complex<double>>>& matrix);

    // Helper function to multiply two matrices
    std::vector<std::vector<std::complex<double>>> multiply(const std::vector<std::vector<std::complex<double>>>& A,
        const std::vector<std::vector<std::complex<double>>>& B);

    // Helper function to compute the covariance matrix
    std::vector<std::vector<std::complex<double>>> compute_covariance(const std::vector<std::complex<double>>& snap);

    // Helper function to perform eigenvalue decomposition manually
    std::pair<std::vector<double>, std::vector<std::vector<std::complex<double>>>> eigen_decomposition(
        std::vector<std::vector<std::complex<double>>>& matrix, int max_iters = 1000, double tol = 1e-6);
}

#endif // DOA_PROCESSING_HPP
