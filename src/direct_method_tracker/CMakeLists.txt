aux_source_directory( . AUX_SRC_DIRECT_METHOD_TRACKER )

# Add all components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility/src )
if ( NOT TARGET lib_slam_utility_datatype )
    add_subdirectory( ${SLAM_UTILITY_PATH}/datatype ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_datatype )
endif()
if ( NOT TARGET lib_slam_utility_math )
    add_subdirectory( ${SLAM_UTILITY_PATH}/math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()
if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()
if ( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()

# Add camera model.
set( SENSOR_CAMERA_MODEL_PATH ${PROJECT_SOURCE_DIR}/../Sensor_Model )
if ( NOT TARGET lib_camera_model )
    add_subdirectory( ${SENSOR_CAMERA_MODEL_PATH}/src/camera ${PROJECT_SOURCE_DIR}/build/lib_camera_model )
endif()

add_library( lib_direct_method_tracker ${AUX_SRC_DIRECT_METHOD_TRACKER} )
target_include_directories( lib_direct_method_tracker PUBLIC
    .
    ..
)
target_link_libraries( lib_direct_method_tracker
    lib_slam_utility_datatype
    lib_slam_utility_math
    lib_slam_utility_operate
    lib_slam_utility_log

    lib_camera_model
)
