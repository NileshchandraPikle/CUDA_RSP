#include "datatypes.hpp"
#include <random>
#include <ctime>
#include <complex>
#include <fstream> // Add this header for file I/O
#include <sstream> // Add this header for string stream operations
#include <iostream> // Add this header for console output


namespace RadarData {
    // Initialize frame with random 16-bit integer values
    Frame initialize_frame(int num_receivers, int num_chirps, int num_samples, int frameIndex) {
        // Create a 3D frame with specified dimensions
        Frame frame(num_receivers,
            std::vector<std::vector<Complex>>(num_chirps,
                std::vector<Complex>(num_samples)));

        // Open the CSV file
        std::ifstream file("/mnt/mydisk/Nilesh/CUDA_RSP/data/radar_indexed.csv");
        if (!file.is_open()) {
            std::cerr << "Error: Could not open radar_indexed.csv" << std::endl;
            return frame;
        }

        // Read and parse the CSV file
        std::string line;
        bool frameDataLoaded = false; // Flag to stop reading once the frame is loaded
        while (std::getline(file, line)) {
            std::istringstream ss(line);
            int frame_number, receiver, chirp, sample;
            double value;

            // Parse the line (frame_number, receiver, chirp, sample, value)
            char delimiter;
            ss >> frame_number >> delimiter >> receiver >> delimiter >> chirp >> delimiter >> sample >> delimiter >> value;

            // Check if the current line corresponds to the requested frame
            if (frame_number == frameIndex) {
                // Check if the indices are within bounds
                if (receiver < num_receivers && chirp < num_chirps && sample < num_samples) {
                    frame[receiver][chirp][sample] = Complex(value, 0); // Store as complex with imaginary part 0
                }
                frameDataLoaded = true; // Mark that we are processing the requested frame
            }
            else if (frameDataLoaded) {
                // Stop reading if we have already loaded the requested frame's data
                break;
            }
        }

        file.close();
        return frame;
    }


    // Calculate frame size in bytes
    size_t frame_size_bytes(const Frame & frame) {
        if (frame.empty() || frame[0].empty() || frame[0][0].empty()) {
            return 0;
        }
        // Each Complex is sizeof(Complex) (16 bytes for double real and imag parts)
        return frame.size() * frame[0].size() * frame[0][0].size() * sizeof(Complex);
    }

}
