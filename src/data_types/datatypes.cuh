#include <cuda_runtime.h>

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
        
        // Constructor to initialize the frame with given dimensions
        Frame(int r, int c, int s);
        
        
        ~Frame();

        inline int idx(int receiver, int chirp, int sample) const {
            return receiver * num_chirps * num_samples + chirp * num_samples + sample;
        }

        Complex& operator()(int receiver, int chirp, int sample);
        const Complex& operator()(int receiver, int chirp, int sample) const;
       
        

        void allocate_frame_mem_device();
        void free_device();
        void copy_frame_to_device();
        void copy_frame_to_host();
        
    };// Frame
    // Function to initialize the frame with random 16-bit integer values
    void  initialize_frame(Frame &frame, int num_receivers, int num_chirps, int num_samples, int frameIndex);

    // Function to calculate frame size in bytes
    size_t frame_size_bytes(const Frame& frame);
    
    // Function to initialize multiple frames for batch processing
    size_t initializeBatchFrames(
        std::vector<Frame>& frames, 
        int numFrames,
        int num_receivers,
        int num_chirps, 
        int num_samples
    );
    
    struct Peak {
        int receiver;
        int chirp;
        int sample;
        double value;
    };// Peak

    struct Target {
        double x, y, z;
        double range;
        double azimuth;
        double elevation;
        double strength;
        double relativeSpeed;
        double rcs;
    };

    class TargetResults {
    public:
        Target* targets; // Host array
        Target* d_targets; // Device array
        int num_targets;

        TargetResults(int max_targets);
        ~TargetResults();
        void allocate_host(int max_targets);
        void allocate_device(int max_targets);
        void free_host();
        void free_device();
        void copy_to_host();
    };
    struct peakInfo {
        int num_receivers;
        int num_chirps;
        int num_samples;
        double value;
        int max_num_peaks;
        int num_peaks;

        double* nci;
        double* foldedNci;
        double* noiseEstimation;
        double* thresholdingMap;
        Peak * peakList;
        
        Complex* peaksnaps;



        double* d_nci;
        double* d_foldedNci;
        double* d_noiseEstimation;
        double* d_thresholdingMap;
        int *d_peak_counter;
        int* d_num_peaks; // Device variable to hold number of peaks
        
        cuDoubleComplex* d_peaksnaps;
        Peak* d_peakList;

        peakInfo(int r, int c, int s);

        ~peakInfo();
        void allocate_peakInfo_mem_host();    
        void allocate_peakInfo_mem_device();        
        
        void copy_peakInfo_to_host();
        
        void free_peakInfo_device();
        void free_peakInfo_host();
        
        void cfar_peak_detection();

        void initializePeakSnaps();
        void freePeakSnaps();
        void copyPeakSnapsToHost();
    };// peakInfo

    struct DoAangles{
        double azimuth;
        double elevation;
    };
    
    struct DoAInfo{
    DoAangles *angles;
    DoAangles *d_angles; // Device pointer for CUDA
    int num_peaks;
    int num_receivers;
    Complex *R;
    cuDoubleComplex *d_R;//Covariance matrix on device
    double* eigenvalues;
    double * eigenvectors;
    double* d_eigenvalues; // Eigenvalues on device
    cuDoubleComplex* d_eigenvectors; // Eigenvectors on device
    cuDoubleComplex* d_eigenvector; // Eigenvector on device
    cuDoubleComplex * d_next_eigenvector; // Next eigenvector on device
    cuDoubleComplex *d_noiseSubspace; // Noise subspace on device
    cuDoubleComplex *d_steeringVector; // Steering vector on device
    

    DoAInfo(int num_peaks, int num_receivers);
    
    ~DoAInfo();
    void allocate_angles_mem_host();
    void allocate_angles_mem_device();
    void free_angles_device();
    void free_angles_host();

    void copy_angles_to_host();

    void allocate_R_mem_host();
    void allocate_R_mem_device();
    void free_R_host();
    void free_R_device();
    
    void copy_R_to_host();
    
    void init_eigenData();
    void free_eigenData();
    void copy_eigenData_to_host();

    void initialize();
    
    void init_noiseSubspace();
    void free_noiseSubspace();
    void init_steeringVector();
    void free_steeringVector();
};// DoAInfo
   
  struct EgoEstimationOutput {
        double* d_sum;
        int* d_count;
        EgoEstimationOutput();
        ~EgoEstimationOutput();
        void allocate();
        void free();
        void zero(cudaStream_t stream = 0);
        void copy_to_host(double& h_sum, int& h_count, cudaStream_t stream = 0) const;
    };//EgoEstimationOutput

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
        bool cleanupFrame = true,
        bool cleanupPeakInfo = true
    );
    
    // Overload for reference parameters (for backward compatibility)
    void cleanupRadarResources(
        Frame& frame,
        peakInfo& peakinfo,
        DoAInfo& doaInfo,
        TargetResults& targetResults
    );
    
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
    );
}

#endif // DATA_TYPES_H