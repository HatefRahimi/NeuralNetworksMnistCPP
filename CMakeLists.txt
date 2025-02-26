cmake_minimum_required(VERSION 3.10)

# Project name
project(NeuralNetworksCPP)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories (for header files)
include_directories(src)

# Add executable for your main file
add_executable(ReadASingleFile src/ReadASingleFile.cpp)
