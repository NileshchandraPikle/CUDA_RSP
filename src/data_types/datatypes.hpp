#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <vector>
#include <complex> 
#include <cstdint> // Include for int16_t
#include <tuple> // Include for std::tuple

namespace RadarData {
    // Define Real as a 16-bit integer
    using Real = double;
	using Complex = std::complex<double>;
    // Define Frame as a 3D vector: receivers x chirps x samples
    using Frame = std::vector<std::vector<std::vector<Complex>>>;
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