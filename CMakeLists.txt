cmake_minimum_required( VERSION 3.0 )

project( optical_flow )

set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall -lpthread" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 20 )

# Add slam utility log and tick_tock.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()
if( NOT TARGET lib_slam_utility_tick_tock )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/tick_tock ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_tick_tock )
endif()

# Add visualizor.
set( VISUALIZOR_PATH ${PROJECT_SOURCE_DIR}/../Visualizor2D )
if( NOT TARGET lib_2d_visualizor )
    add_subdirectory( ${VISUALIZOR_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_2d_visualizor )
endif()

# Add feature point/line detector and feature descriptor.
set( FEATURE_DETECTOR_PATH ${PROJECT_SOURCE_DIR}/../Feature_Detector )
if( NOT TARGET lib_feature_point_detector )
    add_subdirectory( ${FEATURE_DETECTOR_PATH}/src/feature_point_detector ${PROJECT_SOURCE_DIR}/build/lib_feature_point_detector )
endif()
if( NOT TARGET lib_feature_line_detector )
    add_subdirectory( ${FEATURE_DETECTOR_PATH}/src/feature_line_detector ${PROJECT_SOURCE_DIR}/build/lib_feature_line_detector )
endif()
if( NOT TARGET lib_feature_descriptor )
    add_subdirectory( ${FEATURE_DETECTOR_PATH}/src/feature_descriptor ${PROJECT_SOURCE_DIR}/build/lib_feature_descriptor )
endif()
if( NOT TARGET lib_nn_feature_point_detector )
    add_subdirectory( ${FEATURE_DETECTOR_PATH}/src/nn_feature_point_detector ${PROJECT_SOURCE_DIR}/build/lib_nn_feature_point_detector )
endif()

# Add descriptor matcher.
if( NOT TARGET lib_descriptor_matcher )
    add_subdirectory( src/descriptor_matcher ${PROJECT_SOURCE_DIR}/build/lib_descriptor_matcher )
endif()

# Add direct method tracker.
if( NOT TARGET lib_direct_method_tracker )
    add_subdirectory( src/direct_method_tracker ${PROJECT_SOURCE_DIR}/build/lib_direct_method_tracker )
endif()

# Add optical flow tracker.
if( NOT TARGET lib_optical_flow_tracker )
    add_subdirectory( src/optical_flow_tracker ${PROJECT_SOURCE_DIR}/build/lib_optical_flow_tracker )
endif()

# Create executable target to test optical flow.
add_executable( test_optical_flow
    test/test_optical_flow.cpp
)
target_link_libraries( test_optical_flow
    lib_feature_point_detector
    lib_optical_flow_tracker
    lib_slam_utility_log
    lib_slam_utility_tick_tock
    lib_2d_visualizor
)

# Create executable target to test direct method.
add_executable( test_direct_method
    test/test_direct_method.cpp
)
target_link_libraries( test_direct_method
    lib_direct_method_tracker
    lib_slam_utility_log
    lib_slam_utility_tick_tock
    lib_2d_visualizor
)

# Create executable target to test descriptor matcher on brief.
add_executable( test_descriptor_matcher_brief
    test/test_descriptor_matcher_brief.cpp
)
target_link_libraries( test_descriptor_matcher_brief
    lib_feature_point_detector
    lib_feature_descriptor
    lib_descriptor_matcher
    lib_slam_utility_log
    lib_slam_utility_tick_tock
    lib_2d_visualizor
)

# Create executable target to test descriptor matcher on xfeat.
add_executable( test_descriptor_matcher_xfeat
    test/test_descriptor_matcher_xfeat.cpp
)
target_link_libraries( test_descriptor_matcher_xfeat
    lib_nn_feature_point_detector
    lib_descriptor_matcher
    lib_slam_utility_log
    lib_slam_utility_tick_tock
    lib_2d_visualizor
)
