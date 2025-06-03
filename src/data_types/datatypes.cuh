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
        
    };
    // Function to initialize the frame with random 16-bit integer values
    void  initialize_frame(Frame &frame, int num_receivers, int num_chirps, int num_samples, int frameIndex);

    // Function to calculate frame size in bytes
    size_t frame_size_bytes(const Frame& frame);
    
    struct Peak {
        int receiver;
        int chirp;
        int sample;
        double value;
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
    };
    
}

#endif // DATA_TYPES_H