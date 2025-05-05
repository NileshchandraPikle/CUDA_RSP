add_library(target STATIC target_processing.cpp)
target_include_directories(target PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 
target_include_directories(target PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../data_types) 
target_include_directories(target PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../config) 