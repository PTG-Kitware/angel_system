cmake_minimum_required( VERSION 3.5 )
project( ptg_datahub )

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

set( headers_public )
set( sources
  # Nodes
  src/image_converter.cpp
  src/ptg_datahub.cpp
  )
add_library( ptg_datahub SHARED
  ${headers_public} ${sources} )
ament_target_dependencies(ptg_datahub
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  OpenCV
  angel_msgs
  )
install( TARGETS ptg_datahub
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )

# Component registration
rclcpp_components_register_node( ptg_datahub
  PLUGIN "image_converter::ImageConverter"
  EXECUTABLE ImageConverter )
rclcpp_components_register_node( ptg_datahub
  PLUGIN "ptg_datahub::PTGDataHub"
  EXECUTABLE PTGDataHub )

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
