cmake_minimum_required( VERSION 3.10 )

project( optical_flow )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall -lpthread" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 17 )

find_package( OpenCV REQUIRED )

# Add feature point/line detector and feature descriptor.
set( FEATURE_DETECTOR_PATH ${PROJECT_SOURCE_DIR}/../Feature_Detector )
if ( NOT TARGET lib_feature_point_detector )
add_subdirectory( ${FEATURE_DETECTOR_PATH}/src/feature_point_detector ${PROJECT_SOURCE_DIR}/build/lib_feature_point_detector )
endif()
if ( NOT TARGET lib_feature_line_detector )
add_subdirectory( ${FEATURE_DETECTOR_PATH}/src/feature_line_detector ${PROJECT_SOURCE_DIR}/build/lib_feature_line_detector )
endif()
if ( NOT TARGET lib_feature_descriptor )
add_subdirectory( ${FEATURE_DETECTOR_PATH}/src/feature_descriptor ${PROJECT_SOURCE_DIR}/build/lib_feature_descriptor )
endif()

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

# Create executable target to test optical flow.
add_executable( test_optical_flow
    test/test_optical_flow.cpp
)
target_link_libraries( test_optical_flow
    ${OpenCV_LIBS}
    lib_optical_flow_tracker
)

# Create executable target to test direct method.
add_executable( test_direct_method
    test/test_direct_method.cpp
)
target_link_libraries( test_direct_method
    ${OpenCV_LIBS}
    lib_direct_method_tracker
)

# Create executable target to test descriptor matcher.
add_executable( test_descriptor_matcher
    test/test_descriptor_matcher.cpp
)
target_link_libraries( test_descriptor_matcher
    ${OpenCV_LIBS}
    lib_feature_point_detector
    lib_feature_descriptor
    lib_descriptor_matcher
)
