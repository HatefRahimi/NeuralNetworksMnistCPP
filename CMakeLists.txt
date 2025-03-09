cmake_minimum_required(VERSION 3.10)

# Project name
project(NeuralNetworksCPP)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories (for header files)
include_directories(src)

# Add executable for your main file
add_executable(ReadASingleFile src/test_single_image.cpp
        src/test_single_label.cpp
        src/test_data.cpp
        src/data.hpp
        src/test_labels.cpp
        src/labels.hpp
        src/CrossEntropyLoss.hpp
        src/FullyConnected.hpp
        src/Relu.hpp
        src/Softmax.hpp
        src/NeuralNetwork.hpp
        src/test_neural_network.cpp
)
add_executable(mnist src/test_neural_network.cpp)
target_compile_features(mnist PRIVATE cxx_std_20)
target_compile_options(mnist PRIVATE -Wall -Wextra -pedantic -Werror -Wno-unused-parameter -Wno-unused-variable -pg -g -O3)
target_link_options(mnist PRIVATE -pg)