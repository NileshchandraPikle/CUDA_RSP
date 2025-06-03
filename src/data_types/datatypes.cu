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
    allocate_frame_mem_device();
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
void Frame::allocate_frame_mem_device() {
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
void Frame::copy_frame_to_device() {
    size_t total = num_receivers * num_chirps * num_samples;
    CUDA_CHECK(cudaMemcpy(
    d_data,
    reinterpret_cast<const cuDoubleComplex*>(data),
    total * sizeof(cuDoubleComplex),
    cudaMemcpyHostToDevice));
    //std::cout << "Frame Data copied to device" << std::endl;
}
void Frame::copy_frame_to_host() {
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
void initialize_frame(Frame& frame, int num_receivers, int num_chirps, int num_samples, int frameIndex) {
    //Frame frame(num_receivers, num_chirps, num_samples);

    std::ifstream file("/mnt/mydisk/Nilesh/CUDA_RSP/data/radar_indexed.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open radar_indexed.csv" << std::endl;
        return;
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
    //return frame;
}

size_t frame_size_bytes(const Frame& frame) {
    return static_cast<size_t>(frame.num_receivers) *
           frame.num_chirps *
           frame.num_samples *
           sizeof(Complex);
}
peakInfo::peakInfo(int r, int c, int s)
{
    num_receivers = r;
    num_chirps = c;
    num_samples = s;
    std::cout << "Creating peakInfo with dimensions: "
              << num_receivers << " receivers, "
              << num_chirps << " chirps, "
              << num_samples << " samples." << std::endl;
    value = 0.0;
    num_peaks = 0; // Initialize number of peaks to zero
     // Device variable to hold number of peaks
    max_num_peaks = num_receivers*num_chirps*num_samples; // Default value, can be adjusted as needed
    
    nci = nullptr;
    foldedNci = nullptr;
    noiseEstimation = nullptr;
    thresholdingMap = nullptr;
    peakList = nullptr;
    peaksnaps = nullptr;    
    
    d_nci = nullptr;
    d_foldedNci = nullptr;
    d_noiseEstimation = nullptr;
    d_thresholdingMap = nullptr;
    d_peakList = nullptr;
    d_num_peaks = nullptr;
    d_peak_counter = nullptr;
    d_peaksnaps = nullptr;
    allocate_peakInfo_mem_host();
    allocate_peakInfo_mem_device();
}
peakInfo::~peakInfo() {
    free_peakInfo_host();
    free_peakInfo_device();
}
void peakInfo::allocate_peakInfo_mem_host() {
    int size = num_chirps * num_samples;
    //std::cout << "Allocating memory for peakInfo on host: " << size << " elements." << std::endl;
    if (!nci) {
        nci = new double[size];
        memset(nci, 0, size * sizeof(double));
    }
    if (!foldedNci) {
        foldedNci = new double[size];
        memset(foldedNci, 0, size * sizeof(double));
    }
    if (!noiseEstimation) {
        noiseEstimation = new double[size];
        memset(noiseEstimation, 0, size * sizeof(double));
    }
    if (!thresholdingMap) {
        thresholdingMap = new double[size];
        memset(thresholdingMap, 0, size * sizeof(double));
    }
    if (!peakList) {
        peakList = new Peak[max_num_peaks];
        memset(peakList, 0, max_num_peaks * sizeof(Peak));
    }
} //allocate_peakInfo_mem_host
void peakInfo::free_peakInfo_host() {
    delete[] nci;
    delete[] foldedNci;
    delete[] noiseEstimation;
    delete[] thresholdingMap;
    delete[] peakList;

    nci = nullptr;
    foldedNci = nullptr;
    noiseEstimation = nullptr;
    thresholdingMap = nullptr;
    peakList = nullptr;
}// free_peakInfo_host

void peakInfo::allocate_peakInfo_mem_device() {
    int size = num_chirps * num_samples;
    if(!d_peak_counter){
        CUDA_CHECK(cudaMalloc(&d_peak_counter, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_peak_counter, 0, sizeof(int)));
    }
    if (!d_nci) {
        CUDA_CHECK(cudaMalloc(&d_nci, size * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_nci, 0, size * sizeof(double)));
    }
    if (!d_foldedNci) {
        CUDA_CHECK(cudaMalloc(&d_foldedNci, size * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_foldedNci, 0, size * sizeof(double)));
    }
    if (!d_noiseEstimation) {
        CUDA_CHECK(cudaMalloc(&d_noiseEstimation, size * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_noiseEstimation, 0, size * sizeof(double)));
    }
    if (!d_thresholdingMap) {
        CUDA_CHECK(cudaMalloc(&d_thresholdingMap, size * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_thresholdingMap, 0, size * sizeof(double)));
    }
    if (!d_peakList) {
        CUDA_CHECK(cudaMalloc(&d_peakList, max_num_peaks * sizeof(Peak)));
        CUDA_CHECK(cudaMemset(d_peakList, 0, max_num_peaks * sizeof(Peak)));
    }
}// allocate_peakInfo_mem_device
void peakInfo::free_peakInfo_device() {
    if(d_peak_counter) {
        CUDA_CHECK(cudaFree(d_peak_counter));
        d_peak_counter = nullptr;
    }
    if (d_nci) {
        CUDA_CHECK(cudaFree(d_nci));
        d_nci = nullptr;
    }
    if (d_foldedNci) {
        CUDA_CHECK(cudaFree(d_foldedNci));
        d_foldedNci = nullptr;
    }
    if (d_noiseEstimation) {
        CUDA_CHECK(cudaFree(d_noiseEstimation));
        d_noiseEstimation = nullptr;
    }
    if (d_thresholdingMap) {
        CUDA_CHECK(cudaFree(d_thresholdingMap));
        d_thresholdingMap = nullptr;
    }
    if (d_peakList) {
        CUDA_CHECK(cudaFree(d_peakList));
        d_peakList = nullptr;
    }
}//free_peakInfo_device
void peakInfo::copy_peakInfo_to_host() {
    int size = num_chirps * num_samples;
    if(d_peak_counter) {
        CUDA_CHECK(cudaMemcpy(&num_peaks, d_peak_counter, sizeof(int), cudaMemcpyDeviceToHost));
    }
    if (d_nci) {
        CUDA_CHECK(cudaMemcpy(nci, d_nci, size* sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (d_foldedNci) {
        CUDA_CHECK(cudaMemcpy(foldedNci, d_foldedNci, size * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (d_noiseEstimation) {
        CUDA_CHECK(cudaMemcpy(noiseEstimation, d_noiseEstimation, size * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (d_thresholdingMap) {
        CUDA_CHECK(cudaMemcpy(thresholdingMap, d_thresholdingMap, size * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (d_peakList) {
        CUDA_CHECK(cudaMemcpy(peakList, d_peakList, max_num_peaks * sizeof(Peak), cudaMemcpyDeviceToHost));
    }
}
void peakInfo::initializePeakSnaps(){
    if(!peaksnaps)
    {
        peaksnaps = new Complex[num_peaks*num_receivers];
        memset(peaksnaps, 0, num_peaks * num_receivers * sizeof(Complex));
    }
    if(!d_peaksnaps) {
        CUDA_CHECK(cudaMalloc(&d_peaksnaps, num_peaks * num_receivers * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMemset(d_peaksnaps, 0, num_peaks * num_receivers * sizeof(cuDoubleComplex)));
    }
}
void peakInfo::freePeakSnaps() {
    if (peaksnaps) {
        delete[] peaksnaps;
        peaksnaps = nullptr;
    }
    if (d_peaksnaps) {
        CUDA_CHECK(cudaFree(d_peaksnaps));
        d_peaksnaps = nullptr;
    }
} // freePeakSnaps
void peakInfo::copyPeakSnapsToHost() {
    if (d_peaksnaps) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<cuDoubleComplex*>(peaksnaps), d_peaksnaps, num_peaks * num_receivers * sizeof(Complex), cudaMemcpyDeviceToHost));
    }
} // copyPeakSnapsToHost

} // namespace RadarData