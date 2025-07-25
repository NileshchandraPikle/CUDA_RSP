cmake_minimum_required(VERSION 3.18)

project(rsp LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Building both single frame and batch processing versions
message(STATUS "Building both single frame (rsp) and batch processing (rsp_batch) executables")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/config)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/data_types)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_utils)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/preprocessing)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/peak_detection)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/mimo_synthesis)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/doa_processing)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/target_processing)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/rcs)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/ego_estimation)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/ghost_removal)

# Create both executables
message(STATUS "Building both single frame (rsp) and batch processing (rsp_batch) executables")
add_executable(rsp src/main.cu)         # Single frame version with all pipeline stages
add_executable(rsp_batch src/main_batch.cu)  # Batch processing version - only FFT and Peak Detection

# Link libraries to both executables
target_link_libraries(rsp PUBLIC 
                      config 
                      datatypes
                      cuda_utils
                      preprocessed
                      peakdetection
                      mimo
                      doa
                      rcs
                      ego
                      target
                      ghost)

target_link_libraries(rsp_batch PUBLIC 
                      config 
                      datatypes
                      cuda_utils
                      preprocessed
                      peakdetection
                      mimo
                      doa
                      rcs
                      ego
                      target
                      ghost)

