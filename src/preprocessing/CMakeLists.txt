
add_library(preprocessed STATIC fft_processing.cpp)
target_include_directories(preprocessed PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(preprocessed PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../config)  
target_include_directories(preprocessed PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../data_types) 