cmake_minimum_required(VERSION 3.10)

# Project name
project(NeuralNetworksCPP)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories (for header files)
include_directories(src)

# Add executable for your main file
add_executable(mnist src/test_neural_network.cpp)
target_compile_features(mnist PRIVATE cxx_std_20)
target_compile_options(mnist PRIVATE -Wall -Wextra -pedantic -Werror -Wno-unused-parameter -Wno-unused-variable -pg -g -O3)
target_link_options(mnist PRIVATE -pg)

add_executable(test_data src/test_data.cpp)
target_compile_features(test_data PRIVATE cxx_std_20)
target_compile_options(test_data PRIVATE -Wall -Wextra -pedantic -Werror -Wno-unused-parameter -Wno-unused-variable -pg -g)
target_link_options(test_data PRIVATE -pg)

add_executable(test_labels src/test_labels.cpp)
target_compile_features(test_labels PRIVATE cxx_std_20)
target_compile_options(test_labels PRIVATE -Wall -Wextra -pedantic -Werror -Wno-unused-parameter -Wno-unused-variable -pg -g)
target_link_options(test_labels PRIVATE -pg)
