cmake_minimum_required(VERSION 3.8)
project(net_ft_calibration)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/identify_payload_interactive.py
  scripts/gravity_compensation_node.py
  scripts/wrench_frame_transform_node.py
  scripts/set_payload_profile.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# Calibration runs go to the repo's (gitignored) verification/logs, not the install space
get_filename_component(CALIB_LOG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../verification/logs" ABSOLUTE)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/log_dir.txt "${CALIB_LOG_DIR}\n")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/log_dir.txt
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_payload_identification test/test_payload_identification.py)
endif()

ament_package()
