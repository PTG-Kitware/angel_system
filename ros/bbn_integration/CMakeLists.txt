cmake_minimum_required( VERSION 3.5 )
project( bbn_integration )

# Default to C99
if( NOT CMAKE_C_STANDARD )
  set( CMAKE_C_STANDARD 99 )
endif()

# Default to C++14
if( NOT CMAKE_CXX_STANDARD )
  set( CMAKE_CXX_STANDARD 14 )
endif()

if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
  add_compile_options( -Wall -Wextra -Wpedantic )
endif()

# -----------------------------------------------------------------------------
# find dependencies
find_package( ament_cmake REQUIRED )
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package( rclcpp REQUIRED )
find_package( rclcpp_components REQUIRED )
find_package( bbn_integration_msgs REQUIRED )
find_package( yaml-cpp REQUIRED )
# libZMQ does not come with CMake config files, but it does ship with
# pkg-config stuff, so use that.
find_package( PkgConfig REQUIRED )
pkg_check_modules( libzmq REQUIRED libzmq )

# -----------------------------------------------------------------------------
# Integration library
set( headers_public )
set( sources
  # Nodes
  src/nodes/zmq_integration_client.cxx
  )
add_library( bbn_integration SHARED
  ${headers_public}
  ${sources}
  )
ament_target_dependencies( bbn_integration
  rclcpp
  rclcpp_components
  bbn_integration_msgs
  yaml-cpp
  libzmq
  )
# If we have local header files.
#target_include_directories( bbn_integration PUBLIC
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#  $<INSTALL_INTERFACE:include> )
install( TARGETS bbn_integration
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )

# -----------------------------------------------------------------------------
# Component registration
rclcpp_components_register_node( bbn_integration
  PLUGIN "angel::bbn_integration::ZmqIntegrationClient"
  EXECUTABLE ZmqIntegrationClient )

# -----------------------------------------------------------------------------
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
