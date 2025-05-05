cmake_minimum_required(VERSION 3.10)

project(rsp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/config)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/data_types)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/preprocessing)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/peak_detection)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/mimo_synthesis)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/doa_processing)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/target_processing)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/rcs)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/ego_estimation)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/ghost_removal)
add_executable(rsp src/main.cpp)
target_link_libraries(rsp PUBLIC 
                      config 
                      datatypes 
                      preprocessed 
                      peakdetection
                      mimo
                      doa
                      target
                      rcs
                      ego
                      ghost
                      )

