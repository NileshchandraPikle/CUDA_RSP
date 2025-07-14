set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 11)

add_library(rcs STATIC rcs.cu)
target_include_directories(rcs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 
target_include_directories(rcs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../config) 