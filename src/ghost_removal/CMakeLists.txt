set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 11)

add_library(ghost STATIC ghost_removal.cu)
target_include_directories(ghost PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../config
    ${CMAKE_CURRENT_SOURCE_DIR}/../data_types
    ${CMAKE_CURRENT_SOURCE_DIR}/../target_processing
)
target_link_libraries(ghost PUBLIC datatypes target)