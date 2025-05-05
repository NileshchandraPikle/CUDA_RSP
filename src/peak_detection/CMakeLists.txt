
add_library(peakdetection STATIC peak_detection.cpp)
target_include_directories(peakdetection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(peakdetection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../config)   