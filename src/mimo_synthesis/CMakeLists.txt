set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 11)

add_library(mimo STATIC mimo_synthesis.cu)
target_include_directories(mimo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 
target_include_directories(preprocessed PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../data_types) 