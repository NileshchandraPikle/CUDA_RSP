#include "doa_processing.hpp"
#include "../config/config.hpp"
#include <cmath>
#include <iostream>
#include <vector>
#include <complex>
#include <algorithm> // For std::sort
#include <numeric>   // For std::inner_product

namespace DOAProcessing {
    using namespace std;

    // Helper function to compute the Hermitian (conjugate transpose) of a matrix
    vector<vector<complex<double>>> hermitian(const vector<vector<complex<double>>>& matrix) {
        size_t rows = matrix.size();
        size_t cols = matrix[0].size();
        vector<vector<complex<double>>> result(cols, vector<complex<double>>(rows));

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[j][i] = conj(matrix[i][j]);
            }
        }
        return result;
    }

    // Helper function to multiply two matrices
    vector<vector<complex<double>>> multiply(const vector<vector<complex<double>>>& A,
        const vector<vector<complex<double>>>& B) {
        size_t rows = A.size();
        size_t cols = B[0].size();
        size_t inner = B.size();
        vector<vector<complex<double>>> result(rows, vector<complex<double>>(cols, { 0.0, 0.0 }));

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                for (size_t k = 0; k < inner; ++k) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }

    // Helper function to compute the covariance matrix
    vector<vector<complex<double>>> compute_covariance(const vector<complex<double>>& snap) {
        size_t num_receivers = snap.size();
        vector<vector<complex<double>>> R(num_receivers, vector<complex<double>>(num_receivers, { 0.0, 0.0 }));

        for (size_t i = 0; i < num_receivers; ++i) {
            for (size_t j = 0; j < num_receivers; ++j) {
                R[i][j] = snap[i] * conj(snap[j]);
            }
        }
        return R;
    }

    // Helper function to perform eigenvalue decomposition manually
    pair<vector<double>, vector<vector<complex<double>>>> eigen_decomposition(
        vector<vector<complex<double>>>& matrix, int max_iters, double tol) {
        size_t n = matrix.size();
        vector<double> eigenvalues(n, 0.0);
        vector<vector<complex<double>>> eigenvectors(n, vector<complex<double>>(n, { 0.0, 0.0 }));

        for (size_t k = 0; k < n; ++k) {
            vector<complex<double>> eigenvector(n, { 1.0, 0.0 }); // Initial guess
            double eigenvalue = 0.0;

            for (int iter = 0; iter < max_iters; ++iter) {
                // Multiply matrix by the current eigenvector
                vector<complex<double>> next_vector(n, { 0.0, 0.0 });
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        next_vector[i] += matrix[i][j] * eigenvector[j];
                    }
                }

                // Normalize the resulting vector
                double norm = 0.0;
                for (const auto& val : next_vector) {
                    norm += std::norm(val);
                }
                norm = sqrt(norm);
                for (auto& val : next_vector) {
                    val /= norm;
                }

                // Compute the Rayleigh quotient for the eigenvalue
                double next_eigenvalue = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    next_eigenvalue += real(conj(next_vector[i]) * eigenvector[i]);
                }

                // Check for convergence
                if (abs(next_eigenvalue - eigenvalue) < tol) {
                    eigenvalue = next_eigenvalue;
                    eigenvector = next_vector;
                    break;
                }

                eigenvalue = next_eigenvalue;
                eigenvector = next_vector;
            }

            eigenvalues[k] = eigenvalue;
            eigenvectors[k] = eigenvector;

            // Deflate the matrix
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    matrix[i][j] -= eigenvalue * eigenvector[i] * conj(eigenvector[j]);
                }
            }
        }

        return make_pair(eigenvalues, eigenvectors);
    }

    void compute_music_doa(const RadarData::PeakSnaps& peakSnaps,
        vector<pair<double, double>>& doaResults,
        int num_sources) {
        doaResults.clear();

        // Radar parameters
        double wavelength = RadarConfig::WAVELENGTH;
        double d = RadarConfig::ANTENNA_SPACING;

        // Iterate over each peak snap
        for (const auto& snap : peakSnaps) {
            int num_receivers = snap.size();
            if (num_receivers < num_sources) {
                cerr << "Insufficient receivers for MUSIC algorithm." << endl;
                continue;
            }

            // Compute the covariance matrix
            auto R = compute_covariance(snap);

            // Perform eigenvalue decomposition
            pair<vector<double>, vector<vector<complex<double>>>> eigen_result =
                eigen_decomposition(R);
            vector<double> eigenvalues = eigen_result.first;
            vector<vector<complex<double>>> eigenvectors = eigen_result.second;

            // Separate signal and noise subspaces
            vector<vector<complex<double>>> noiseSubspace;
            for (int i = num_sources; i < num_receivers; ++i) {
                noiseSubspace.push_back(eigenvectors[i]);
            }

            // MUSIC spectrum calculation
            double azimuth = 0.0, elevation = 0.0;
            double max_spectrum = -1.0;

            for (double theta = -90.0; theta <= 90.0; theta += 1.0) {
                for (double phi = -90.0; phi <= 90.0; phi += 1.0) {
                    // Steering vector
                    vector<complex<double>> steering(num_receivers);
                    for (int i = 0; i < num_receivers; ++i) {
                        double phase = 2.0 * RadarConfig::PI * d * i *
                            (sin(theta * RadarConfig::PI / 180.0) *
                                cos(phi * RadarConfig::PI / 180.0)) / wavelength;
                        steering[i] = exp(complex<double>(0, phase));
                    }

                    // Compute MUSIC spectrum
                    double spectrum = 0.0;
                    for (const auto& noiseVec : noiseSubspace) {
                        double projection = std::abs(std::inner_product(noiseVec.begin(), noiseVec.end(),
                            steering.begin(), std::complex<double>(0, 0)));
                        spectrum += 1.0 / (projection * projection);
                    }

                    if (spectrum > max_spectrum) {
                        max_spectrum = spectrum;
                        azimuth = theta;
                        elevation = phi;
                    }
                }
            }

            // Store the result
            doaResults.emplace_back(azimuth, elevation);
        }
    }
}
