#ifndef CONFIG_HPP
#define CONFIG_HPP

namespace RadarConfig {
    // Default radar parameters (compile-time constants)
    constexpr int NUM_RECEIVERS = 3;        // Number of receivers (R)
    constexpr int NUM_TRANSMITTERS = 1;     // Number of transmit antennas (TX)
    constexpr int NUM_CHIRPS = 128;        // Number of chirps per frame (C)
    constexpr int NUM_SAMPLES = 256;       // Number of samples per chirp (S)
    constexpr double WAVELENGTH = 0.03;     // Wavelength in meters (lambda)
    constexpr double ANTENNA_SPACING = WAVELENGTH / 2.0; // Antenna spacing in meters (d)
    constexpr int SAMPLE_SIZE_BYTES = 2;    // Size of one sample in bytes (real + imaginary)

    constexpr double PI = 3.14159265359;    // Mathematical constant Pi
	constexpr int TRAINING_CELLS = 10; // Number of training cells for CFAR
	constexpr int GUARD_CELLS = 2; // Number of guard cells for CFAR
	constexpr double FALSE_ALARM_RATE = 0.01; // False alarm rate for CFAR
    // Runtime-configurable parameters
    struct Config {
        int num_receivers;        // Number of receivers
        int num_transmitters;     // Number of transmit antennas
        int num_chirps;           // Number of chirps
        int num_samples;          // Number of samples
        double wavelength;        // Wavelength in meters
        double antenna_spacing;   // Antenna spacing in meters

        // Default constructor initializes with compile-time constants
        Config()
            : num_receivers(NUM_RECEIVERS),
            num_transmitters(NUM_TRANSMITTERS),
            num_chirps(NUM_CHIRPS),
            num_samples(NUM_SAMPLES),
            wavelength(WAVELENGTH),
            antenna_spacing(ANTENNA_SPACING) {
        }
    };
    // Function to load configuration (implemented in config.cpp)
    Config load_config();
}
#endif // CONFIG_H