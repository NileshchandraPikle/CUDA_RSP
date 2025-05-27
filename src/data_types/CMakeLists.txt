set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 11)


add_library(datatypes STATIC datatypes.cu)
target_include_directories(datatypes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 