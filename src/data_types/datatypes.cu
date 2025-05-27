#include "datatypes.cuh"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include "../cuda_utils/cuda_utils.hpp"

namespace RadarData {

Frame::Frame(int r, int c, int s)
    : num_receivers(r), num_chirps(c), num_samples(s), d_data(nullptr)
{
    data = new Complex[r * c * s]();
}

Frame::~Frame() {
    delete[] data;
    free_device();
}

Complex& Frame::operator()(int receiver, int chirp, int sample) {
    return data[idx(receiver, chirp, sample)];
}
const Complex& Frame::operator()(int receiver, int chirp, int sample) const {
    return data[idx(receiver, chirp, sample)];
}

// Device memory management
void Frame::allocate_device() {
    if (!d_data) {
        size_t total = num_receivers * num_chirps * num_samples;
        CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(cuDoubleComplex)));
    }
}
void Frame::free_device() {
    if (d_data) {
        CUDA_CHECK(cudaFree(d_data));
        d_data = nullptr;
    }
}
void Frame::copy_to_device() {
    allocate_device();
    size_t total = num_receivers * num_chirps * num_samples;
    CUDA_CHECK(cudaMemcpy(
    d_data,
    reinterpret_cast<const cuDoubleComplex*>(data),
    total * sizeof(cuDoubleComplex),
    cudaMemcpyHostToDevice));
    std::cout << "Frame Data copied to device" << std::endl;
}
void Frame::copy_to_host() {
    if (d_data) {
        size_t total = num_receivers * num_chirps * num_samples;
       CUDA_CHECK(cudaMemcpy(
    reinterpret_cast<cuDoubleComplex*>(data),
    d_data,
    total * sizeof(cuDoubleComplex),
    cudaMemcpyDeviceToHost));
    }
}

// Initialize frame with data from CSV
Frame initialize_frame(int num_receivers, int num_chirps, int num_samples, int frameIndex) {
    Frame frame(num_receivers, num_chirps, num_samples);

    std::ifstream file("/mnt/mydisk/Nilesh/CUDA_RSP/data/radar_indexed.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open radar_indexed.csv" << std::endl;
        return frame;
    }

    std::string line;
    bool frameDataLoaded = false;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        int frame_number, receiver, chirp, sample;
        double value;
        char delimiter;
        ss >> frame_number >> delimiter >> receiver >> delimiter >> chirp >> delimiter >> sample >> delimiter >> value;

        if (frame_number == frameIndex) {
            if (receiver < num_receivers && chirp < num_chirps && sample < num_samples) {
                frame(receiver, chirp, sample) = Complex(value, 0);
            }
            frameDataLoaded = true;
        } else if (frameDataLoaded) {
            break;
        }
    }
    file.close();
    return frame;
}

size_t frame_size_bytes(const Frame& frame) {
    return static_cast<size_t>(frame.num_receivers) *
           frame.num_chirps *
           frame.num_samples *
           sizeof(Complex);
}

} // namespace RadarData