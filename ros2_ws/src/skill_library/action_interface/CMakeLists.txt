cmake_minimum_required(VERSION 3.8)
project(action_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

find_package(rosidl_default_generators REQUIRED)

# Each skill's own action definition gets one more line here -- e.g. once
# fine_alignment is merged in, add "action/FineAlignment.action" below.
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/AlignPort.action"
  "action/FineAlignment.action"
)

ament_package()
