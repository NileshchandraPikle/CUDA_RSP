add_library(rcs STATIC rcs.cpp)
target_include_directories(rcs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 
target_include_directories(rcs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../config) 