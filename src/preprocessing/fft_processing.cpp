#include <vector>
#include <complex>
#include <iostream>
#include "fft_processing.hpp"
#include "../data_types/datatypes.hpp"
#include "../config/config.hpp"

namespace fftProcessing {
    // Iterative FFT implementation
    void fft(std::vector<std::complex<double>>& data, bool inverse) {
        const size_t N = data.size();
        if (N <= 1) return; // Base case
        // Bit reversal permutation
        size_t j = 0;
        for (size_t i = 1; i < N; ++i) {
            size_t bit = N >> 1;
            while (j >= bit) {
                j -= bit;
                bit >>= 1;
            }
            j += bit;
            if (i < j) std::swap(data[i], data[j]);
        }
        // Cooley-Tukey FFT
        for (size_t len = 2; len <= N; len <<= 1) {
            double angle = 2 * RadarConfig::PI / len * (inverse ? -1 : 1);
            std::complex<double> wlen(cos(angle), sin(angle));
            for (size_t i = 0; i < N; i += len) {
                std::complex<double> w(1);
                for (size_t j = 0; j < len / 2; ++j) {
                    std::complex<double> u = data[i + j];
                    std::complex<double> t = w * data[i + j + len / 2];
                    data[i + j] = u + t;
                    data[i + j + len / 2] = u - t;
                    w *= wlen;
                }
            }
        }
        if (inverse) {
            for (size_t i = 0; i < N; ++i) {
                data[i] /= N; // Normalize the result
            }
        }
    }

    // Apply Hilbert transform to the samples dimension
    void apply_hilbert_transform_samples(RadarData::Frame& frame) {
        size_t num_receivers = frame.size();
        size_t num_chirps = frame[0].size();
        size_t num_samples = frame[0][0].size();
        // Check if the frame is empty
        if (num_receivers == 0 || num_chirps == 0 || num_samples == 0) {
            std::cout << "Frame is empty, nothing to process." << std::endl;
            return; // Nothing to process
        }
        for (int r = 0; r < num_receivers; r++) {
            for (int c = 0; c < num_chirps; c++) {
                std::vector<std::complex<double>> data(num_samples);
                // Copy data to data vector by converting each sample to complex with imaginary part 0
                for (int s = 0; s < num_samples; s++) {
                    data[s] = frame[r][c][s]; // Already complex with imaginary part 0
                }
                // Apply FFT to the data vector to get frequency domain representation
                fft(data, false);
                // Apply Hilbert transform in frequency domain
                for (int s = 1; s < num_samples / 2; s++) {
                    data[s] *= 2; // Double the amplitude of the positive frequencies
                }
                for (int s = num_samples / 2; s < num_samples; s++) {
                    data[s] = 0; // Set the negative frequencies to zero
                }
                fft(data, true); // Apply inverse FFT to get back to time domain
                // Copy the data back to the frame
                for (int s = 0; s < num_samples; s++) {
                    frame[r][c][s] = data[s]; // Copy the complex value back to the frame
                }
            }
        }
    }

  

    // Apply FFT1 to the frame
    void apply_fft1(RadarData::Frame& frame) {
        int num_receivers = frame.size();
        int num_chirps = frame[0].size();
        int num_samples = frame[0][0].size();
        // Check if the frame is empty
        if (num_receivers == 0 || num_chirps == 0 || num_samples == 0) {
            std::cout << "Frame is empty, nothing to process." << std::endl;
            return; // Nothing to process
        }
        for (int r = 0; r < num_receivers; r++) {
            for (int c = 0; c < num_chirps; c++) {
                // Perform FFT directly on the frame data
                std::vector<std::complex<double>>& data = frame[r][c];
                fft(data, true); // Apply FFT to the data vector
            }
        }
    }

     // Function to apply Hanning window
    void apply_hanning_window(std::vector<std::complex<double>>& data) {
        size_t N = data.size();
        for (size_t n = 0; n < N; ++n) {
            double window_value = 0.5 * (1 - cos(2 * RadarConfig::PI * static_cast<double>(n) / static_cast<double>(N - 1)));

            data[n] *= window_value;
        }
    }
    // Normalize fft output
    void normalize_fft_output(std::vector<std::complex<double>>& data, size_t fft_length) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] /= fft_length;
        }
    }
    // Apply FFT2 to the frame
    void apply_fft2(RadarData::Frame& frame) {
        size_t num_receivers = frame.size();
        size_t num_chirps = frame[0].size();
        size_t num_samples = frame[0][0].size();
        // Check if the frame is empty
        if (num_receivers == 0 || num_chirps == 0 || num_samples == 0) {
            std::cout << "Frame is empty, nothing to process." << std::endl;
            return; // Nothing to process
        }

        for (int r = 0; r < num_receivers; r++) {
            for (int s = 0; s < num_samples; s++) {
                std::vector<std::complex<double>> data(num_chirps);
                // Copy data to data vector by converting each sample to complex with imaginary part 0
                for (int c = 0; c < num_chirps; c++) {
                    data[c] = frame[r][c][s]; // Already complex with imaginary part 0
                }
                // Apply Hanning window
                apply_hanning_window(data);
                fft(data, false); // Apply FFT to the data vector
                normalize_fft_output(data, num_chirps);
                for (int c = 0; c < num_chirps; c++) {
                    frame[r][c][s] = data[c]; // Copy the complex value back to the frame
                }
            }
        }
    }

    // Function to process the frame with Hilbert transform, FFT1, and FFT2
    void fftProcessPipeline(RadarData::Frame& frame) {
        // Apply Hilbert transform on the sample dimension
        apply_hilbert_transform_samples(frame);
        
        // Apply FFT1 on the sample dimension
        apply_fft1(frame);
        // Apply FFT2 on the chirp dimension
        apply_fft2(frame);
    }
}
