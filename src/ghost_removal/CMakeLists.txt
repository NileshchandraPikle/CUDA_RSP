add_library(ghost STATIC ghost_removal.cpp)
target_include_directories(ghost PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 
target_include_directories(ghost PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../config) 