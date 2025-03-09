#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#define EPSILON 1e-12

class CrossEntropyLoss {
private:
    Eigen::MatrixXd predTensorCache;
public:
    CrossEntropyLoss() {}
    ~CrossEntropyLoss() {}

    // Cross-entropy loss
    double forward(const Eigen::MatrixXd &inputTensor, const Eigen::MatrixXd &labelTensor) {
        predTensorCache = inputTensor;


        // Calculate cross-entropy loss
        double loss = -((labelTensor.array() * (inputTensor.array().log())).sum());
        return loss;
    }

    // Gradient for backpropagation
    Eigen::MatrixXd backward(const Eigen::MatrixXd &labelTensor) {

        Eigen::MatrixXd gradient = - (labelTensor.array() / predTensorCache.array());
        return gradient;
    }
};