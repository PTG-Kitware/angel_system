cmake_minimum_required( VERSION 3.5 )
project( angel_debug )

# Default to C++14
if( NOT CMAKE_CXX_STANDARD )
  set( CMAKE_CXX_STANDARD 14 )
endif()

if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
  add_compile_options( -Wall -Wextra -Wpedantic )
endif()

# find dependencies
find_package( ament_cmake REQUIRED )
find_package( rclcpp REQUIRED )
find_package( rclcpp_components REQUIRED )
find_package( sensor_msgs REQUIRED )
find_package( cv_bridge REQUIRED )
find_package( OpenCV REQUIRED COMPONENTS core )
find_package( angel_msgs REQUIRED )
find_package( angel_utils REQUIRED )

#
# angel_debug Library
#
set( headers_public )
set( sources
  # Nodes
  angel_debug/nodes/simple_2d_detection_overlay.cxx
  )
add_library( angel_debug SHARED
  ${headers_public} ${sources} )
ament_target_dependencies( angel_debug
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  OpenCV
  angel_msgs
  angel_utils
  )
install( TARGETS angel_debug
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )

# Component registration
rclcpp_components_register_node( angel_debug
  PLUGIN "angel_debug::Simple2dDetectionOverlay"
  EXECUTABLE Simple2dDetectionOverlay )

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
