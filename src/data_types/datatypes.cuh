#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <vector>
#include <complex> 
#include <cuComplex.h> // Include for cuDoubleComplex
#include <cstdint> // Include for int16_t
#include <tuple> // Include for std::tuple

namespace RadarData {
    // Define Real as a 16-bit integer
    using Real = double;
	using Complex = std::complex<double>;
    
    // Define Frame as a 3D vector: receivers x chirps x samples
   struct Frame {
        Complex* data; // Flattened 1D array
        int num_receivers;
        int num_chirps;
        int num_samples;
        cuDoubleComplex* d_data; // Device pointer for CUDA

        Frame(int r, int c, int s);
        
        
        ~Frame();

        inline int idx(int receiver, int chirp, int sample) const {
            return receiver * num_chirps * num_samples + chirp * num_samples + sample;
        }

        Complex& operator()(int receiver, int chirp, int sample);
        const Complex& operator()(int receiver, int chirp, int sample) const;
        void allocate_device();
        void free_device();
        void copy_to_device();
        void copy_to_host();
        
    };
    // Function to initialize the frame with random 16-bit integer values
    Frame initialize_frame(int num_receivers, int num_chirps, int num_samples, int frameIndex);

    // Function to calculate frame size in bytes
    size_t frame_size_bytes(const Frame& frame);

    // Define NCI, folded NCI, noise estimation, thresholding map, and Peak List
    using NCI = std::vector<std::vector<Real>>;
    using FoldedNCI = std::vector<std::vector<Real>>;
    using NoiseEstimation = std::vector<std::vector<Real>>;
    using ThresholdingMap = std::vector<std::vector<Real>>;
    using PeakList = std::vector<std::tuple<int, int, int>>;
    using PeakSnaps = std::vector<std::vector<std::complex<double>>>;
	using PeakSnap = std::vector<std::complex<double>>;
}

#endif // DATA_TYPES_H