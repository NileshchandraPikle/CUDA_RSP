add_library(doa STATIC doa_processing.cpp)
target_include_directories(doa PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 
target_include_directories(doa PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../data_types) 