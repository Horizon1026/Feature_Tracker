cmake_minimum_required( VERSION 3.10 )

project( OPTICAL_FLOW )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall" )

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )

# include_directories( "/usr/include/eigen3" )
include_directories( ${PROJECT_SOURCE_DIR}/src )
include_directories( ${PROJECT_SOURCE_DIR}/test )

aux_source_directory( ${PROJECT_SOURCE_DIR}/src OPTICAL_FLOW )

add_executable( test_optical_flow
    ${OPTICAL_FLOW}
    test/test_optical_flow.cpp
)

add_library( OPTICAL_FLOW
    ${OPTICAL_FLOW}
)

target_link_libraries( test_optical_flow
    # TBB::tbb
    ${OpenCV_LIBS}
)
