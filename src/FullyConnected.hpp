#include "Eigen/Dense"
#include <iostream>
#include <random>

class FullyConnected
{
private:
    Eigen::MatrixXd weights;
    size_t input_size;
    size_t output_size;
    double range = 1.0;

    Eigen::MatrixXd input_tensor;

public:
    FullyConnected() {
    }

    ~FullyConnected() {}
    FullyConnected(size_t in, size_t out) : input_size(in), output_size(out)
    {

        range = 1.0 / sqrt(input_size);
        weights = Eigen::MatrixXd::Random(input_size+1, output_size);
        weights = weights * range;
    };

    void setWeights(Eigen::MatrixXd w) {
        // for testing purposes
        weights = w;
    }

    Eigen::MatrixXd forward(Eigen::MatrixXd input) {

        input_tensor = Eigen::MatrixXd(input.rows(), input.cols() + 1);
        auto ones = Eigen::MatrixXd::Constant(input.rows(), 1,1.0);
        input_tensor << input, ones;
        Eigen::MatrixXd output= input_tensor*weights;
        return output;
    }

    Eigen::MatrixXd backward(Eigen::MatrixXd error_tensor, double learningRate = 0.001) {

        Eigen::MatrixXd gradient_weights(weights.rows(), weights.cols());
        gradient_weights = input_tensor.transpose() * error_tensor;
        weights -= learningRate * gradient_weights;

        return error_tensor * weights.transpose();
    }

};


