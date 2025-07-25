set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 11)

# Header-only library, so we don't need to build anything
add_library(cuda_utils INTERFACE)
target_include_directories(cuda_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
