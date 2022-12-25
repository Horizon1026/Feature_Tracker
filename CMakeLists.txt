cmake_minimum_required( VERSION 3.10 )

project( klt_tracker )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall" )

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )

# include_directories( "/usr/include/eigen3" )
include_directories( ${PROJECT_SOURCE_DIR}/src )
include_directories( ${PROJECT_SOURCE_DIR}/test )

aux_source_directory( ${PROJECT_SOURCE_DIR}/src KLT_TRACKER )

add_executable( test_klt_tracker
    ${KLT_TRACKER}
    test/test_klt_tracker.cpp
)

add_library( klt_tracker
    ${KLT_TRACKER}
)

target_link_libraries( test_klt_tracker
    # TBB::tbb
    ${OpenCV_LIBS}
)
