# @file
# @author Erwin Meza Vega <emezav@unicauca.edu.co>, <emezav@gmail.com>
# @copyright MIT License
# Configuration file for CMake.

# Minimum required version: 3.24 which improves support for CUDA configuration.
cmake_minimum_required(VERSION 3.24)

# Get current directory name and replace spaces with underscores.
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME ProjectName)
string(REPLACE " " "_" ProjectId ${ProjectName})

# Use current directory as project name (Visual Studio solution name)
project(${ProjectName}
  VERSION 1.0
  LANGUAGES CXX CUDA)

# To use a custom project name:
# comment the project(...) line before this comment,
# replace MYPROJECT with the desired project name and uncomment the following line:
#project("MYPROJECT")

# Enable CUDA language is deprecated on CMake 3.8 onwards - use LANGUAGES on project(...)
# See https://developer.nvidia.com/blog/building-cuda-applications-cmake/
#enable_language(CUDA)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Set position independent code to ON, required for some CUDA libraries
# Please check carefully for each library required on the project
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use folders to group CMake targets
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Add c/c++ headers and sources
file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/include/*.h")

# Add CUDA headers and sources
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu" "${CMAKE_SOURCE_DIR}/include/*.cuh")

# Add include directory "./include"
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add include directory for the CUDA include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Add executable to the project
add_executable (${PROJECT_NAME} ${cpu_source_files} ${gpu_source_files})

# Add CUDA::* (depending on project requirements) to link libraries
#target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart <..other cuda libraries..>)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

# Set C++ to std 17
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# Enable separable compilation (Required for some libraries)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set default start project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# Create compile_commands.json
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# Set link flags to mirror project link flags
set_property(TARGET ${PROJECT_NAME} PROPERTY LINK_FLAGS ${PROJ_LINK_FLAGS})

# Set CUDA architecture to native (use host architecture) on current project
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES native)

# Set debugger working directory to source dir
set_property(TARGET ${PROJECT_NAME}  PROPERTY DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Set debugger working directory to source dir (for Visual Studio)
set_property(TARGET ${PROJECT_NAME}  PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
