cmake_minimum_required( VERSION 3.10 )

project( optical_flow )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall -lpthread" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

find_package( OpenCV REQUIRED )

# Add slam utility
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if ( NOT TARGET lib_slam_utility )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_slam_utility )
endif()

# Add optical flow tracker
if ( NOT TARGET lib_optical_flow_tracker )
    add_subdirectory( src ${PROJECT_SOURCE_DIR}/build/lib_optical_flow_tracker )
endif()

add_executable( test_optical_flow
    test/test_optical_flow.cpp
)
target_link_libraries( test_optical_flow
    ${OpenCV_LIBS}
    lib_slam_utility
    lib_optical_flow_tracker
)

