#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

/**
 * @file cuda_utils.hpp
 * @brief Utility macros and functions for CUDA error handling
 */

/**
 * @brief Macro to check CUDA function calls for errors
 * 
 * This macro wraps CUDA calls and checks for errors. If an error occurs,
 * it throws a runtime_error exception instead of exiting the program,
 * allowing for graceful error handling.
 * 
 * @param call The CUDA function call to check
 * @throws std::runtime_error if a CUDA error occurs
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::string error_msg = "CUDA error at " + std::string(__FILE__) + ":" + \
                                   std::to_string(__LINE__) + " - " + \
                                   std::string(cudaGetErrorString(err)); \
            std::cerr << error_msg << std::endl; \
            throw std::runtime_error(error_msg); \
        } \
    } while (0)

/**
 * @brief Checks for any pending CUDA errors
 * 
 * @param message Optional message to include with the error
 * @throws std::runtime_error if a CUDA error is pending
 */
inline void checkCudaErrors(const char* message = "") {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = std::string(message) + " CUDA error: " + 
                               std::string(cudaGetErrorString(err));
        std::cerr << error_msg << std::endl;
        throw std::runtime_error(error_msg);
    }
}
/**
 * @brief Template class for CUDA resource management using RAII pattern
 * 
 * This template helps manage CUDA resources by automatically
 * deallocating them when the object goes out of scope.
 * 
 * @tparam T The type of CUDA resource
 * @tparam Deleter The function type used to deallocate the resource
 */
template<typename T, void(*Deleter)(T*)>
class CudaResourceManager {
private:
    T* resource;
    bool owns_resource;

public:
    /**
     * @brief Constructor that takes ownership of a resource
     * 
     * @param res Pointer to the CUDA resource
     */
    explicit CudaResourceManager(T* res) : resource(res), owns_resource(true) {}
    
    /**
     * @brief Deleted copy constructor to prevent double-free issues
     */
    CudaResourceManager(const CudaResourceManager&) = delete;
    
    /**
     * @brief Move constructor
     */
    CudaResourceManager(CudaResourceManager&& other) noexcept 
        : resource(other.resource), owns_resource(other.owns_resource) {
        other.owns_resource = false;
    }
    
    /**
     * @brief Deleted assignment operator
     */
    CudaResourceManager& operator=(const CudaResourceManager&) = delete;
    
    /**
     * @brief Destructor that frees the resource if owned
     */
    ~CudaResourceManager() {
        if (owns_resource && resource) {
            Deleter(resource);
        }
    }
    
    /**
     * @brief Access the managed resource
     * 
     * @return Pointer to the managed resource
     */
    T* get() const { return resource; }
};

/**
 * @brief Helper function for CUDA device memory allocation
 * 
 * @tparam T Type to allocate
 * @param count Number of elements to allocate
 * @return Pointer to allocated memory
 * @throws std::runtime_error if allocation fails
 */
template<typename T>
T* cudaAllocateDevice(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

#endif // CUDA_UTILS_HPP