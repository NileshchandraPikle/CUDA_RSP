add_library(ego STATIC ego_estimation.cpp)
target_include_directories(ego PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 
target_include_directories(ego PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../config) 