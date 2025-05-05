add_library(mimo STATIC mimo_synthesis.cpp)
target_include_directories(mimo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 
target_include_directories(preprocessed PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../data_types) 