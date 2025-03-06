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

        Eigen::MatrixXd clipped_predicted = inputTensor.array().max(EPSILON).min(1 - EPSILON);

        // Calculate cross-entropy loss
        double loss = -((labelTensor.array() * clipped_predicted.array().log()).sum());
        return loss;
    }

    // Gradient for backpropagation
    Eigen::MatrixXd backward(const Eigen::MatrixXd &labelTensor) {

        Eigen::MatrixXd gradient = - (labelTensor.array() / predTensorCache.array());
        return gradient;
    }
};