#pragma once

#include <iostream>
#include <Eigen/Dense>
#include "FullyConnected.hpp"


// Example usage
int main() {
    FullyConnectedLayer fcLayer(784, 10); // For MNIST-like input (784 pixels) to 10 classes

    // Simulated input: a batch of 2 images, each with 784 pixels
    Eigen::MatrixXd input(2, 784);
    input.setRandom(); // Random values to simulate pixel data

    // Forward pass to get the output
    Eigen::MatrixXd output = fcLayer.forward(input);
    std::cout << "Output of Forward Pass:\n" << output << std::endl;

    // Simulated gradients from the next layer (e.g., from cross-entropy loss)
    Eigen::MatrixXd gradients(2, 10);
    gradients.setRandom();

    // Backward pass to adjust weights and compute input gradients
    Eigen::MatrixXd inputGradients = fcLayer.backward(gradients);
    std::cout << "Input Gradients:\n" << inputGradients << std::endl;

    return 0;
}
//
// Created by hatef on 3/6/2025.
//
