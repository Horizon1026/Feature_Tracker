cmake_minimum_required( VERSION 3.10 )

project( optical_flow )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall -lpthread" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 17 )

find_package( OpenCV REQUIRED )

# Add descriptor matcher.
if ( NOT TARGET lib_descriptor_matcher )
    add_subdirectory( src/descriptor_matcher ${PROJECT_SOURCE_DIR}/build/lib_descriptor_matcher )
endif()

# Add direct method tracker.
if ( NOT TARGET lib_direct_method_tracker )
    add_subdirectory( src/direct_method_tracker ${PROJECT_SOURCE_DIR}/build/lib_direct_method_tracker )
endif()

# Add optical flow tracker.
if ( NOT TARGET lib_optical_flow_tracker )
    add_subdirectory( src/optical_flow_tracker ${PROJECT_SOURCE_DIR}/build/lib_optical_flow_tracker )
endif()

add_executable( test_optical_flow
    test/test_optical_flow.cpp
)
target_link_libraries( test_optical_flow
    ${OpenCV_LIBS}
    lib_optical_flow_tracker
)
