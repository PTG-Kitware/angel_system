ament_python_install_package( "${PROJECT_NAME}" )

install(PROGRAMS
  angel_utils/bag_extractor.py
  DESTINATION lib/${PROJECT_NAME}
)
