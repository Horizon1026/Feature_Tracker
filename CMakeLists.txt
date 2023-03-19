cmake_minimum_required( VERSION 3.10 )

project( optical_flow )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )

include_directories( ${PROJECT_SOURCE_DIR}/src )
aux_source_directory( ${PROJECT_SOURCE_DIR}/src OPTICAL_FLOW )

# Add slam utility
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if ( NOT TARGET lib_slam_utility )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_slam_utility )
endif()

add_executable( test_optical_flow
    ${OPTICAL_FLOW}
    test/test_optical_flow.cpp
)
target_link_libraries( test_optical_flow
    ${OpenCV_LIBS}
    lib_slam_utility
)

add_executable( test_lk_greftools
    ${OPTICAL_FLOW}
    test/test_lk_greftools.cpp
)
target_link_libraries( test_lk_greftools
    ${OpenCV_LIBS}
    tcmalloc profiler
    lib_slam_utility
)
