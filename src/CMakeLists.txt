aux_source_directory( . AUX_SRC_OPTICAL_FLOW_TRACKER )

if ( NOT TARGET lib_slam_utility_datatype )
    add_subdirectory( datatype ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_datatype )
endif()

if ( NOT TARGET lib_slam_utility_math )
    add_subdirectory( math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()

if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()

add_library( lib_optical_flow_tracker ${AUX_SRC_OPTICAL_FLOW_TRACKER} )
target_include_directories( lib_optical_flow_tracker PUBLIC . )
target_link_libraries( lib_optical_flow_tracker
    lib_slam_utility_datatype
    lib_slam_utility_math
    lib_slam_utility_operate
)
