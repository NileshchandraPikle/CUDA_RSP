#include "doa_processing.cuh"
#include "../data_types/datatypes.cuh"
#include "../config/config.hpp"
#include "../cuda_utils/cuda_utils.hpp"
#include <cmath>
#include <iostream>
#include <vector>
#include <complex>
#include <algorithm> // For std::sort
#include <numeric>   // For std::inner_product
#include <stdexcept> // For std::runtime_error

namespace DOAProcessing {
    using namespace std;
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
    try {
        // Parameter validation
        if (peakinfo.d_peaksnaps == nullptr) {
            throw std::runtime_error("Invalid peak snapshots data (null pointer)");
        }
        
        if (peakinfo.num_peaks <= 0) {
            std::cout << "No peaks to process in DoA estimation" << std::endl;
            return;
        }
        
        if (num_sources <= 0 || num_sources >= doAInfo.num_receivers) {
            throw std::runtime_error("Invalid number of sources for MUSIC algorithm: " + 
                                    std::to_string(num_sources) + 
                                    " (must be positive and less than number of receivers: " + 
                                    std::to_string(doAInfo.num_receivers) + ")");
        }
        
        int num_receivers = doAInfo.num_receivers;
        
        // Calculate kernel launch parameters
        const int threads_per_block = 256;
        dim3 threads(threads_per_block, 1, 1);
        dim3 blocks((peakinfo.num_peaks + threads.x - 1) / threads.x, 1, 1);
        
        std::cout << "DoA Processing: Estimating angles for " << peakinfo.num_peaks 
                  << " peaks using MUSIC algorithm with " << num_sources << " sources" << std::endl;
        std::cout << "  Launch parameters: " << blocks.x << " blocks, " 
                  << threads_per_block << " threads per block" << std::endl;
        
        // Launch kernel
        compute_music_doa_kernel<<<blocks, threads>>>(
            peakinfo.d_peaksnaps, 
            peakinfo.num_peaks, 
            doAInfo.d_R, 
            num_receivers, 
            num_sources,
            doAInfo.d_eigenvalues,
            doAInfo.d_eigenvector, 
            doAInfo.d_eigenvectors, 
            doAInfo.d_next_eigenvector, 
            doAInfo.d_noiseSubspace, 
            doAInfo.d_angles);
        
        // Wait for kernel to complete
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Check for any kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        
        // Copy results back to host
        doAInfo.copy_angles_to_host();
        
        // Print DoA results (uncomment for debugging)
        /*
        for(int i = 0; i < peakinfo.num_peaks; ++i) {
            std::cout << "Peak " << i + 1 << ": Azimuth = " 
                    << doAInfo.angles[i].azimuth << "°, Elevation = " 
                    << doAInfo.angles[i].elevation << "°" << std::endl;
        }
        */
        
    } catch (const std::exception& e) {
        std::cerr << "Error in DoA processing: " << e.what() << std::endl;
        throw;
    }
    }
}
