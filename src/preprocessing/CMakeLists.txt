set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 11)

add_library(preprocessed STATIC fft_processing.cu)
target_include_directories(preprocessed PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(preprocessed PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../config)  
target_include_directories(preprocessed PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../data_types) 