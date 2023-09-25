cmake_minimum_required( VERSION 3.5 )
project( angel_msgs )

# Default to C++14
if( NOT CMAKE_CXX_STANDARD )
  set( CMAKE_CXX_STANDARD 14 )
endif()

if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
  add_compile_options( -Wall -Wextra -Wpedantic )
endif()

# find dependencies
find_package( ament_cmake REQUIRED )
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package( rosidl_default_generators REQUIRED )
find_package( builtin_interfaces REQUIRED )
find_package( std_msgs REQUIRED )
find_package( geometry_msgs REQUIRED )
find_package( shape_msgs REQUIRED )
find_package( vision_msgs REQUIRED )

set( message_files
  msg/ActivityDetection.msg
  msg/AnnotationEvent.msg
  msg/AruiObject3d.msg
  msg/AruiUpdate.msg
  msg/AruiUserNotification.msg
  msg/EyeGazeData.msg
  msg/HandJointPose.msg
  msg/HandJointPosesUpdate.msg
  msg/HeadsetAudioData.msg
  msg/HeadsetPoseData.msg
  msg/InterpretedAudioUserEmotion.msg
  msg/InterpretedAudioUserIntent.msg
  msg/ObjectDetection2d.msg
  msg/ObjectDetection2dSet.msg
  msg/ObjectDetection3dSet.msg
  msg/SpatialMesh.msg
  msg/SystemCommands.msg
  msg/TaskGraph.msg
  msg/TaskItem.msg
  msg/TaskNode.msg
  msg/TaskUpdate.msg
  msg/Utterance.msg
  msg/VisionBoundingBox3d.msg
  )
set( service_files
  srv/QueryAllObjects3d.srv
  srv/QueryTaskGraph.srv
  srv/QueryImageSize.srv
  )
rosidl_generate_interfaces( ${PROJECT_NAME} ${message_files} ${service_files}
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs shape_msgs vision_msgs
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
