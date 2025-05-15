set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 11)

add_library(config STATIC config.cpp)
target_include_directories(config PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 