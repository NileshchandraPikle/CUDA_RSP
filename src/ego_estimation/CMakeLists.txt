set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 11)


add_library(ego STATIC ego_estimation.cu)
target_include_directories(ego PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 
target_include_directories(ego PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../config) 