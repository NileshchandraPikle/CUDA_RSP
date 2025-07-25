#include "datatypes.cuh"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include "../cuda_utils/cuda_utils.hpp"

namespace RadarData {
EgoEstimationOutput::EgoEstimationOutput() : d_sum(nullptr), d_count(nullptr) {}
EgoEstimationOutput::~EgoEstimationOutput() { free(); }
void EgoEstimationOutput::allocate() {
    cudaMalloc(&d_sum, sizeof(double));
    cudaMalloc(&d_count, sizeof(int));
}
void EgoEstimationOutput::free() {
    if (d_sum) cudaFree(d_sum);
    if (d_count) cudaFree(d_count);
    d_sum = nullptr;
    d_count = nullptr;
}
void EgoEstimationOutput::zero(cudaStream_t stream) {
    cudaMemsetAsync(d_sum, 0, sizeof(double), stream);
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);
}
void EgoEstimationOutput::copy_to_host(double& h_sum, int& h_count, cudaStream_t stream) const {
    cudaMemcpyAsync(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

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

// Initialize multiple frames for batch processing
size_t initializeBatchFrames(
    std::vector<Frame>& frames, 
    int numFrames,
    int num_receivers,
    int num_chirps, 
    int num_samples
) {
    // Clear the vector in case it has existing frames
    frames.clear();
    frames.reserve(numFrames);
    
    std::cout << "Initializing " << numFrames << " frames for batch processing..." << std::endl;
    
    // Calculate memory requirements
    size_t mem_per_frame = sizeof(std::complex<double>) * num_receivers * num_chirps * num_samples;
    size_t total_mem = mem_per_frame * numFrames;
    std::cout << "Memory per frame: " << mem_per_frame / (1024*1024) << " MB" << std::endl;
    std::cout << "Total memory for batch: " << total_mem / (1024*1024) << " MB" << std::endl;
    
    // Check if we have enough GPU memory
    size_t free_mem = 0, total_mem_gpu = 0;
    cudaMemGetInfo(&free_mem, &total_mem_gpu);
    std::cout << "GPU memory: " << total_mem_gpu / (1024*1024) << " MB total, " 
              << free_mem / (1024*1024) << " MB free" << std::endl;
    
    try {
        for (int i = 0; i < numFrames; ++i) {
            std::cout << "Initializing frame " << i+1 << "/" << numFrames << "..." << std::endl;
            
            // Create a new frame
            frames.emplace_back(num_receivers, num_chirps, num_samples);
            
            // Initialize frame with data
            initialize_frame(
                frames.back(),
                num_receivers,
                num_chirps,
                num_samples,
                i % numFrames // Cycle through available frame indices
            );
            
            std::cout << "Copying frame " << i+1 << " to device..." << std::endl;
            // Copy to device
            frames.back().copy_frame_to_device();
            std::cout << "Frame " << i+1 << " initialized and copied to device." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during frame initialization: " << e.what() << std::endl;
        throw;
    }
    
    // Calculate and return frame size in bytes for a single frame
    size_t frame_size = frame_size_bytes(frames[0]);
    std::cout << "Single frame size: " << frame_size << " bytes" << std::endl;
    std::cout << "Total batch size: " << (frame_size * numFrames) / (1024*1024) << " MB" << std::endl;
    
    return frame_size * numFrames;
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




DoAInfo::DoAInfo(int num_peaks, int num_receivers)
    : num_peaks(num_peaks), num_receivers(num_receivers), angles(nullptr), d_angles(nullptr), d_R(nullptr), 
      d_eigenvectors(nullptr), d_eigenvalues(nullptr), d_eigenvector(nullptr), d_next_eigenvector(nullptr), 
      d_noiseSubspace(nullptr), d_steeringVector(nullptr), R(nullptr), eigenvalues(nullptr), eigenvectors(nullptr){
    // Initialize any required resources
    initialize();
    
}
DoAInfo::~DoAInfo() {
    free_angles_host();
    free_angles_device();
    free_R_device();
    free_eigenData();
    free_noiseSubspace();
    free_steeringVector();
    free_R_host();
}

void DoAInfo::allocate_angles_mem_host() {
    if (!angles) {
        angles = new DoAangles[num_peaks];
        memset(angles, 0, num_peaks * sizeof(DoAangles));
    }
} // allocate_angles_mem_host

void DoAInfo::free_angles_host() {
    delete[] angles;
    angles = nullptr;
} // free_angles_host
void DoAInfo::allocate_angles_mem_device() {
    if (!d_angles) {
        CUDA_CHECK(cudaMalloc(&d_angles, num_peaks * sizeof(DoAangles)));
        CUDA_CHECK(cudaMemset(d_angles, 0, num_peaks * sizeof(DoAangles)));
    }
} // allocate_angles_mem_device
void DoAInfo::free_angles_device() {
    if (d_angles) {
        CUDA_CHECK(cudaFree(d_angles));
        d_angles = nullptr;
    }
} // free_angles_device
void DoAInfo::copy_angles_to_host() {
    if (d_angles) {
        CUDA_CHECK(cudaMemcpy(angles, d_angles, num_peaks * sizeof(DoAangles), cudaMemcpyDeviceToHost));
    }
} // copy_angles_to_host
void DoAInfo::initialize() {
    allocate_angles_mem_host();
    allocate_angles_mem_device();
    allocate_R_mem_device();    
    init_eigenData();
    init_noiseSubspace();
    init_steeringVector();
    allocate_R_mem_host();
} // initialize

void DoAInfo::allocate_R_mem_device() {
    if (!d_R) {
        size_t size = num_peaks*num_receivers * num_receivers * sizeof(cuDoubleComplex);
        CUDA_CHECK(cudaMalloc(&d_R, size));
        CUDA_CHECK(cudaMemset(d_R, 0, size));
    }

} // allocate_R_mem_device
void DoAInfo::free_R_device() {
    if (d_R) {
        CUDA_CHECK(cudaFree(d_R));
        d_R = nullptr;
    }
} // free_R_device

void DoAInfo::init_eigenData() {
    if (!d_eigenvectors) {
        size_t size = num_peaks*num_receivers * num_receivers * sizeof(cuDoubleComplex);
        CUDA_CHECK(cudaMalloc(&d_eigenvectors, size));
        CUDA_CHECK(cudaMemset(d_eigenvectors, 0, size));
    }
    if (!d_eigenvalues) {
        CUDA_CHECK(cudaMalloc(&d_eigenvalues, num_peaks*num_receivers * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_eigenvalues, 0, num_peaks*num_receivers * sizeof(double)));
    }
    if (!d_eigenvector) {
        CUDA_CHECK(cudaMalloc(&d_eigenvector, num_peaks*num_receivers * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMemset(d_eigenvector, 0, num_peaks*num_receivers * sizeof(cuDoubleComplex)));
    }
    if (!d_next_eigenvector) {
        CUDA_CHECK(cudaMalloc(&d_next_eigenvector, num_peaks*num_receivers * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMemset(d_next_eigenvector, 0, num_peaks*num_receivers * sizeof(cuDoubleComplex)));
    }
    if(!eigenvalues) {
        eigenvalues = new double[num_peaks*num_receivers];
        memset(eigenvalues, 0, num_peaks*num_receivers * sizeof(double));
    }
    if(!eigenvectors) {
        eigenvectors = new double[num_peaks*num_receivers * num_receivers];
        memset(eigenvectors, 0, num_peaks*num_receivers * num_receivers * sizeof(double));
    }
} // init_eigenData
void DoAInfo::free_eigenData() {
    if (d_eigenvectors) {
        CUDA_CHECK(cudaFree(d_eigenvectors));
        d_eigenvectors = nullptr;
    }
    if (d_eigenvalues) {
        CUDA_CHECK(cudaFree(d_eigenvalues));
        d_eigenvalues = nullptr;
    }
    if (d_eigenvector) {
        CUDA_CHECK(cudaFree(d_eigenvector));
        d_eigenvector = nullptr;
    }
    if (d_next_eigenvector) {
        CUDA_CHECK(cudaFree(d_next_eigenvector));
        d_next_eigenvector = nullptr;
    }
    if (eigenvalues) {
        delete[] eigenvalues;
        eigenvalues = nullptr;
    }
} // free_eigenData

void DoAInfo::copy_eigenData_to_host() {
    if (d_eigenvectors) {
        size_t size = num_peaks*num_receivers * num_receivers * sizeof(cuDoubleComplex);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<cuDoubleComplex*>(R), d_eigenvectors, size, cudaMemcpyDeviceToHost));
    }
    if (d_eigenvalues) {
        CUDA_CHECK(cudaMemcpy(eigenvalues, d_eigenvalues, num_peaks*num_receivers * sizeof(double), cudaMemcpyDeviceToHost));
    }
} // copy_eigenData_to_host
void DoAInfo::init_noiseSubspace() {
    if (!d_noiseSubspace) {
        size_t size = num_peaks * num_receivers * sizeof(cuDoubleComplex);
        CUDA_CHECK(cudaMalloc(&d_noiseSubspace, size));
        CUDA_CHECK(cudaMemset(d_noiseSubspace, 0, size));
    }
} // init_noiseSubspace
void DoAInfo::free_noiseSubspace() {
    if (d_noiseSubspace) {
        CUDA_CHECK(cudaFree(d_noiseSubspace));
        d_noiseSubspace = nullptr;
    }
} // free_noiseSubspace
void DoAInfo::init_steeringVector() {
    if (!d_steeringVector) {
        size_t size = num_peaks * num_receivers * sizeof(cuDoubleComplex);
        CUDA_CHECK(cudaMalloc(&d_steeringVector, size));
        CUDA_CHECK(cudaMemset(d_steeringVector, 0, size));
    }
} // init_steeringVector
void DoAInfo::free_steeringVector() {
    if (d_steeringVector) {
        CUDA_CHECK(cudaFree(d_steeringVector));
        d_steeringVector = nullptr;
    }
} // free_steeringVector
void DoAInfo::copy_R_to_host() {
    if (d_R) {
        size_t size = num_peaks*num_receivers * num_receivers * sizeof(cuDoubleComplex);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<cuDoubleComplex*>(R), d_R, size, cudaMemcpyDeviceToHost));
    }
} // copy_R_to_host
void DoAInfo::allocate_R_mem_host() {
    if (!R) {
        R = new Complex[num_peaks*num_receivers * num_receivers];
        memset(R, 0, num_peaks*num_receivers * num_receivers * sizeof(Complex));
    }
} // allocate_R_mem_host
void DoAInfo::free_R_host() {
    delete[] R;
    R = nullptr;
} // free_R_host

TargetResults::TargetResults(int max_targets)
    : targets(nullptr), d_targets(nullptr), num_targets(max_targets) {
    allocate_host(max_targets);
    allocate_device(max_targets);
}

TargetResults::~TargetResults() {
    free_host();
    free_device();
}

void TargetResults::allocate_host(int max_targets) {
    if (!targets) {
        targets = new Target[max_targets];
        memset(targets, 0, max_targets * sizeof(Target));
        for (int i = 0; i < max_targets; ++i) {
            targets[i].rcs = 0.0;
        }
    }
}

void TargetResults::allocate_device(int max_targets) {
    if (!d_targets) {
        CUDA_CHECK(cudaMalloc(&d_targets, max_targets * sizeof(Target)));
        CUDA_CHECK(cudaMemset(d_targets, 0, max_targets * sizeof(Target)));
        // Optionally set rcs to 0.0 in device memory (not strictly needed after memset)
    }
}

void TargetResults::free_host() {
    if (targets) {
        delete[] targets;
        targets = nullptr;
    }
}

void TargetResults::free_device() {
    if (d_targets) {
        CUDA_CHECK(cudaFree(d_targets));
        d_targets = nullptr;
    }
}

void TargetResults::copy_to_host() {
    if (d_targets && targets) {
        CUDA_CHECK(cudaMemcpy(targets, d_targets, num_targets * sizeof(Target), cudaMemcpyDeviceToHost));
    }
}

/**
 * Free all GPU memory resources associated with radar data structures
 * 
 * This function centralizes all memory cleanup operations for radar data structures
 * to ensure consistent and complete memory management.
 * 
 * @param frame Pointer to radar frame structure to cleanup (nullptr to skip)
 * @param peakinfo Pointer to peak detection information structure to cleanup (nullptr to skip)
 * @param doaInfo Pointer to direction of arrival information structure to cleanup (nullptr to skip)
 * @param targetResults Pointer to target processing results structure to cleanup (nullptr to skip)
 * @param cleanupFrame Whether to clean up frame resources (default: true)
 * @param cleanupPeakInfo Whether to clean up peak info resources (default: true)
 */
void cleanupRadarResources(
    Frame* frame,
    peakInfo* peakinfo,
    DoAInfo* doaInfo,
    TargetResults* targetResults,
    bool cleanupFrame,
    bool cleanupPeakInfo
) {
    // Clean up frame resources if provided and flag is set
    if (frame && cleanupFrame) {
        frame->free_device();
    }
    
    // Clean up peak detection resources if provided and flag is set
    if (peakinfo && cleanupPeakInfo) {
        peakinfo->free_peakInfo_device();
    }
    
    // Clean up DoA processing resources if provided
    if (doaInfo) {
        doaInfo->free_angles_device();
        doaInfo->free_R_device();
        doaInfo->free_eigenData();
        doaInfo->free_noiseSubspace();
        doaInfo->free_steeringVector();
    }
    
    // Clean up target processing resources if provided
    if (targetResults) {
        targetResults->free_device();
    }
    
    // Output memory cleanup confirmation
    std::cout << "Radar processing resources cleaned up successfully" << std::endl;
}

// Overload for reference parameters (for backward compatibility)
void cleanupRadarResources(
    Frame& frame,
    peakInfo& peakinfo,
    DoAInfo& doaInfo,
    TargetResults& targetResults
) {
    cleanupRadarResources(&frame, &peakinfo, &doaInfo, &targetResults, true, true);
}

/**
 * Clean up all resources associated with batch processing
 * 
 * This function centralizes all memory cleanup operations for batch processing,
 * including persistent arrays, frame arrays, and other radar data structures.
 * 
 * @param frames Vector of radar frames to clean up
 * @param peakInfos Vector of peak detection information structures to clean up
 * @param doaInfo Direction of arrival information structure to clean up
 * @param targetResults Target processing results structure to clean up
 * @param persistentArraysInitialized Whether persistent arrays have been initialized
 * @param cleanupPersistentArrays Function pointer to cleanup persistent arrays
 */
void cleanupBatchResources(
    std::vector<Frame>& frames,
    std::vector<peakInfo>& peakInfos,
    DoAInfo& doaInfo,
    TargetResults& targetResults,
    bool persistentArraysInitialized,
    void (*cleanupPersistentArrays)()
) {
    // First, clean up any persistent arrays used for batch processing
    if (persistentArraysInitialized && cleanupPersistentArrays) {
        std::cout << "Cleaning up persistent arrays for batch peak detection..." << std::endl;
        cleanupPersistentArrays();
    }
    
    // Clean up frame resources
    for (auto& frame : frames) {
        frame.free_device();
    }
    
    // Clean up peak info resources
    for (auto& pi : peakInfos) {
        pi.free_peakInfo_device();
        pi.free_peakInfo_host();
    }
    
    // Clean up other shared resources
    doaInfo.free_angles_device();
    doaInfo.free_R_device();
    doaInfo.free_eigenData();
    doaInfo.free_noiseSubspace();
    doaInfo.free_steeringVector();
    
    // Clean up target results
    targetResults.free_device();
    
    std::cout << "Batch processing resources cleaned up successfully" << std::endl;
}

} // namespace RadarData