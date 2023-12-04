cmake_minimum_required( VERSION 3.5 )
project( angel_utils )

# Default to C++14
if( NOT CMAKE_CXX_STANDARD )
  set( CMAKE_CXX_STANDARD 14 )
endif()

if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
  add_compile_options( -Wall -Wextra -Wpedantic )
endif()

# Find required bits
find_package( ament_cmake REQUIRED )
find_package( ament_cmake_python REQUIRED )
find_package( rclcpp REQUIRED )
find_package( rclcpp_components REQUIRED )
find_package( rclpy REQUIRED )
find_package( sensor_msgs REQUIRED )
find_package( OpenCV REQUIRED COMPONENTS core )
find_package( cv_bridge REQUIRED )
find_package( image_transport REQUIRED )
find_package( angel_msgs REQUIRED )

#
# Cpp things
#
# See https://docs.ros.org/en/foxy/How-To-Guides/Ament-CMake-Documentation.html#building-a-library
# for important details that made this work properly, specifically
# `ament_export_targets` and the `EXPORT` part of `install( TARGETS ...`.
# Note that in `install( TARGETS`, "angel_utils" is used both for the
#
add_library( angel_utils SHARED
  src/rate_tracker.cpp
  nodes/simple_2d_detection_overlay.cxx
  )
target_include_directories( angel_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )
ament_target_dependencies( angel_utils
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  image_transport
  OpenCV
  angel_msgs
  )
ament_export_targets( angel_utils_targets HAS_LIBRARY_TARGET )
# Include the following if this lib introduces transitive dependencies, i.e.
# exposes dependencies via public headers (e.,g. OpenCV if including in an hpp).
# ament_export_dependencies( )

# Install Things
install( DIRECTORY include/
  DESTINATION include
  )
install( TARGETS angel_utils
  EXPORT angel_utils_targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
  )

# Component registration
rclcpp_components_register_node( angel_utils
PLUGIN "angel_utils::Simple2dDetectionOverlay"
EXECUTABLE Simple2dDetectionOverlay )

#
# Python things
#
add_subdirectory( python )

install(PROGRAMS
  scripts/bag_extractor.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/bag_extractor.sh
  DESTINATION lib/${PROJECT_NAME}
)

if( BUILD_TESTING )
  find_package( ament_lint_auto REQUIRED )
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
