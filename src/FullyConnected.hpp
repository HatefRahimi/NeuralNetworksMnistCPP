#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <random>

class FullyConnectedLayer {
private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::MatrixXd inputCache;


public:
    FullyConnectedLayer(int inputSize, int outputSize) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, std::sqrt(2.0 / inputSize));

        weights = Eigen::MatrixXd(outputSize, inputSize).unaryExpr([&](double) { return d(gen); });
        biases = Eigen::VectorXd::Zero(outputSize);
    }

    // Forward pass: Compute the output of the fully connected layer
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) {
        inputCache = input;
        // Matrix multiplication with bias addition
        Eigen::MatrixXd output = (input * weights.transpose()).rowwise() + biases.transpose();
        return output;
    }

    // Backward pass: Compute the gradients and return input gradient
    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradients, double learningRate = 0.01) {
        // Calculate gradients with respect to weights and biases
        Eigen::MatrixXd weightGradients = gradients.transpose() * inputCache;
        Eigen::VectorXd biasGradients = gradients.colwise().sum();

        // Calculate gradients with respect to input for backpropagation
        Eigen::MatrixXd inputGradients = gradients * weights;

        weights -= learningRate * weightGradients;
        biases -= learningRate * biasGradients;

        return inputGradients;
    }
};

