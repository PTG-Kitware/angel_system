cmake_minimum_required(VERSION 3.5)
project(ptg_datahub)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(angel_msgs REQUIRED)
find_package(cv_bridge REQUIRED )
find_package(OpenCV REQUIRED COMPONENTS core )

add_executable(datahub src/ptg_datahub.cpp)
ament_target_dependencies(datahub
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  angel_msgs
  shape_msgs
  cv_bridge
  OpenCV
  )

install(TARGETS
  datahub
  DESTINATION lib/${PROJECT_NAME})

ament_package()
