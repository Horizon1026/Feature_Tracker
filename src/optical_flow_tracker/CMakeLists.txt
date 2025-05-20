aux_source_directory( . AUX_SRC_OPTICAL_FLOW_TRACKER )
aux_source_directory( basic_klt AUX_SRC_OPTICAL_FLOW_BASIC_KLT )
aux_source_directory( affine_klt AUX_SRC_OPTICAL_FLOW_AFFINE_KLT )
aux_source_directory( lssd_klt AUX_SRC_OPTICAL_FLOW_LSSD_KLT )

# Add all relative components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if( NOT TARGET lib_slam_utility_basic_type )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/basic_type ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_basic_type )
endif()
if( NOT TARGET lib_slam_utility_math )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()
if( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()
if( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()

# Add all relative components of slam utility data type.
if( NOT TARGET lib_image )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_type/image ${PROJECT_SOURCE_DIR}/build/lib_image )
endif()
if( NOT TARGET lib_image_pyramid )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_type/image_pyramid ${PROJECT_SOURCE_DIR}/build/lib_image_pyramid )
endif()

add_library( lib_optical_flow_tracker
    ${AUX_SRC_OPTICAL_FLOW_TRACKER}
    ${AUX_SRC_OPTICAL_FLOW_BASIC_KLT}
    ${AUX_SRC_OPTICAL_FLOW_AFFINE_KLT}
    ${AUX_SRC_OPTICAL_FLOW_LSSD_KLT}
)
target_include_directories( lib_optical_flow_tracker PUBLIC
    .
    ..
    basic_klt
    affine_klt
    lssd_klt
)
target_link_libraries( lib_optical_flow_tracker
    lib_slam_utility_basic_type
    lib_slam_utility_math
    lib_slam_utility_operate
    lib_slam_utility_log

    lib_image
    lib_image_pyramid
)
