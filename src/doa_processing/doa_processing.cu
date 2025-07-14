#include "doa_processing.cuh"
#include"../data_types/datatypes.cuh"
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
  /*  vector<vector<complex<double>>> compute_covariance(const vector<complex<double>>& snap) {
        size_t num_receivers = snap.size();
        vector<vector<complex<double>>> R(num_receivers, vector<complex<double>>(num_receivers, { 0.0, 0.0 }));

        for (size_t i = 0; i < num_receivers; ++i) {
            for (size_t j = 0; j < num_receivers; ++j) {
                R[i][j] = snap[i] * conj(snap[j]);
            }
        }
        return R;
    }*/
    __device__ void compute_covariance(cuDoubleComplex *snap, int num_receivers, cuDoubleComplex *d_R, int peak_index)
    {       
        for (size_t i = 0; i < num_receivers; ++i) {
            for (size_t j = 0; j < num_receivers; ++j) {
                size_t idx = i * num_receivers + j;
                d_R[peak_index * num_receivers * num_receivers + idx] = cuCmul(snap[i], cuConj(snap[j])); // Store in the correct position for the peak index
            }
        }
    }

    __device__ void eigen_decomposition(cuDoubleComplex *d_R, double *d_eigenvalues, cuDoubleComplex* d_eigenvector, cuDoubleComplex *d_eigenvectors, cuDoubleComplex * d_next_eigenvector, int num_receivers, int num_peaks, int peak_index, int max_iters, double tol) {
        
        for(int k = 0; k < num_receivers; k++)
        {
          d_eigenvector[peak_index * num_receivers + k] = make_cuDoubleComplex(1.0, 0.0); // Initial guess
          double eigenvalue = 0.0;
          for(int itr = 0; itr < max_iters; itr++ )
          {
              for (size_t i = 0; i < num_receivers; ++i) {
                    d_next_eigenvector[peak_index * num_receivers + i] = make_cuDoubleComplex(0.0, 0.0); // Reset before accumulation
                    for (size_t j = 0; j < num_receivers; ++j) {
                        d_next_eigenvector[peak_index * num_receivers + i] = cuCadd(d_next_eigenvector[peak_index * num_receivers + i],cuCmul(d_R[peak_index * num_receivers * num_receivers + i * num_receivers + j], d_eigenvector[peak_index * num_receivers + j]));
                    } // j loop
              }// i loop

              double norm = 0.0;
                for (size_t i = 0; i < num_receivers; ++i) {
                    norm += cuCabs(d_next_eigenvector[peak_index * num_receivers + i]) * cuCabs(d_next_eigenvector[peak_index * num_receivers + i]);
                }
                norm = sqrt(norm);
                for (size_t i = 0; i < num_receivers; ++i) {
                    d_next_eigenvector[peak_index * num_receivers + i] = cuCdiv(d_next_eigenvector[peak_index * num_receivers + i], make_cuDoubleComplex(norm, 0.0));
                }
                double next_eigenvalue = 0.0;
                for (size_t i = 0; i < num_receivers; ++i) {
                    next_eigenvalue += cuCreal(cuCmul(cuConj(d_next_eigenvector[peak_index * num_receivers + i]),d_eigenvector[peak_index * num_receivers + i]));
            }   
                // Check for convergence
                if (abs(next_eigenvalue - eigenvalue) < tol) {
                    eigenvalue = next_eigenvalue;
                    for (size_t i = 0; i < num_receivers; ++i) {
                        d_eigenvector[peak_index * num_receivers + i] = d_next_eigenvector[peak_index * num_receivers + i];
                    }// i loop
                    break;
                } // if bracket
                eigenvalue = next_eigenvalue;
                for (size_t i = 0; i < num_receivers; ++i) {
                    d_eigenvector[peak_index * num_receivers + i] = d_next_eigenvector[peak_index * num_receivers + i];
                } // i loop
          }// iter loop

          
        d_eigenvalues[peak_index * num_receivers + k] = eigenvalue; // Store the eigenvalue
        // Store the k-th eigenvector for this peak
        for (size_t i = 0; i < num_receivers; ++i) {
            d_eigenvectors[peak_index * num_receivers * num_receivers + k * num_receivers + i] = d_eigenvector[peak_index * num_receivers + i];
        }
        // Deflate the matrix
            for (size_t i = 0; i < num_receivers; ++i) {
                for (size_t j = 0; j < num_receivers; ++j) {
                    d_R[peak_index * num_receivers * num_receivers + i * num_receivers + j] = cuCsub(
                        d_R[peak_index * num_receivers * num_receivers + i * num_receivers + j],
                        cuCmul(make_cuDoubleComplex(eigenvalue, 0.0), cuCmul(d_eigenvector[peak_index * num_receivers + i], cuConj(d_eigenvector[peak_index * num_receivers + j]))));
                } // j loop
        }//  k loop 
    }
    
}

__device__ cuDoubleComplex cuCdotc(const cuDoubleComplex* a, const cuDoubleComplex* b, int n) {
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < n; ++i) {
        sum = cuCadd(sum, cuCmul(cuConj(a[i]), b[i]));
    }
    return sum;
}

__global__ void compute_music_doa_kernel(cuDoubleComplex* d_peaksnaps, int num_peaks, 
    cuDoubleComplex *d_R, int num_receivers, int num_sources, double *d_eigenvalues,
    cuDoubleComplex *d_eigenvector, cuDoubleComplex *d_eigenvectors, cuDoubleComplex * d_next_eigenvector, 
    cuDoubleComplex *d_noiseSubspace, RadarData::DoAangles* d_angles, int max_iters, double tol) {
        double d = RadarConfig::ANTENNA_SPACING;
        int peak_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (peak_index >= num_peaks) return;
        compute_covariance(&d_peaksnaps[peak_index*num_receivers],num_receivers,d_R,peak_index);
        eigen_decomposition(d_R, d_eigenvalues,d_eigenvector, d_eigenvectors, d_next_eigenvector, num_receivers, num_peaks,peak_index, max_iters, tol);      
        // Copy noise eigenvectors correctly: each eigenvector is of length num_receivers
        for(int i = num_sources; i < num_receivers; ++i) {
            for(int j = 0; j < num_receivers; ++j) {
                d_noiseSubspace[peak_index * num_receivers * (num_receivers - num_sources) + (i - num_sources) * num_receivers + j] =
                    d_eigenvectors[peak_index * num_receivers * num_receivers + i * num_receivers + j];
            }
        }
        double azimuth = 0.0, elevation = 0.0;
        double max_spectrum = -1.0;

        cuDoubleComplex* steering = (cuDoubleComplex*)malloc(num_receivers * sizeof(cuDoubleComplex));
        if (!steering) return;
        for(double theta = -90.0; theta <= 90.0; theta += 1.0) {
            for (double phi = -90.0; phi <= 90.0; phi += 1.0) {
                for (int i = 0; i < num_receivers; ++i) {
                    double phase = 2.0 * RadarConfig::PI * d * i *
                        (sin(theta * RadarConfig::PI / 180.0) *
                            cos(phi * RadarConfig::PI / 180.0)) / RadarConfig::WAVELENGTH;
                    steering[i] = make_cuDoubleComplex(cos(phase), sin(phase));
                }
                double spectrum = 0.0;
                // Project onto each noise eigenvector
                for (int n = 0; n < (num_receivers - num_sources); ++n) {
                    cuDoubleComplex* noiseVec = d_noiseSubspace + peak_index * num_receivers * (num_receivers - num_sources) + n * num_receivers;
                    double projection = cuCabs(cuCdotc(noiseVec, steering, num_receivers));
                    spectrum += 1.0 / (projection * projection + 1e-12); // add epsilon to avoid div by zero
                }
                if (spectrum > max_spectrum) {
                    max_spectrum = spectrum;
                    azimuth = theta;
                    elevation = phi;
                }
            }
        }
        // Store the result in the DoAInfo structure
        d_angles[peak_index].azimuth = azimuth;
        d_angles[peak_index].elevation = elevation;
        free(steering);
    }//compute_music_doa_kernel

void compute_music_doa(const RadarData::peakInfo& peakinfo,
        RadarData::DoAInfo& doAInfo, int num_sources) {
    int num_receivers = doAInfo.num_receivers;
    dim3 threads(256, 1, 1);
    dim3 blocks((peakinfo.num_peaks + threads.x - 1) / threads.x, 1, 1);
    compute_music_doa_kernel<<<blocks, threads>>>(peakinfo.d_peaksnaps, peakinfo.num_peaks, doAInfo.d_R, num_receivers, num_sources,
        doAInfo.d_eigenvalues,doAInfo.d_eigenvector, doAInfo.d_eigenvectors, doAInfo.d_next_eigenvector, doAInfo.d_noiseSubspace, doAInfo.d_angles);
    cudaDeviceSynchronize();
    doAInfo.copy_angles_to_host();
    for(int i = 0; i < peakinfo.num_peaks; ++i) {
        std::cout << "Peak " << i + 1 << ": Azimuth = " 
        << doAInfo.angles[i].azimuth << ", Elevation = " 
        << doAInfo.angles[i].elevation << std::endl;
    }
    //doAInfo.copy_R_to_host();
        
        /*for(int i = 0; i < peakinfo.num_peaks; ++i) {
            for(int j = 0; j < num_receivers; ++j) {
                for(int k = 0; k < num_receivers; ++k) {
                    std::cout << "R[" << i << "][" << j << "][" << k << "] = " 
                    << doAInfo.R[i * num_receivers * num_receivers + j * num_receivers + k].real() << " + "
                    << doAInfo.R[i * num_receivers * num_receivers + j * num_receivers + k].imag() << "i" << std::endl;
                }
            }
        }// end for loop*/
        //copy eigendata to host
        //doAInfo.copy_eigenData_to_host();
        /*for(int i = 0; i < peakinfo.num_peaks; ++i) {
            for(int j = 0; j < num_receivers; ++j) {
                std::cout << "Eigenvalue[" << i << "][" << j << "] = " 
                << doAInfo.eigenvalues[i * num_receivers + j] << std::endl;
            }
        }// end for loop*/




        
         
       /*
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
        }*/
    }
}
