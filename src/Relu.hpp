
#include "Eigen/Dense"
#include <iostream>

class ReLUActivation {
private:
    Eigen::MatrixXd input_cache;

public:
    ReLUActivation() {}
    ~ReLUActivation() {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) {
        input_cache = input;
        Eigen::MatrixXd output = input.array().max(0.0);
        return output;
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient) {

        Eigen::MatrixXd mask = (input_cache.array() >= 0.0).cast<double>();
        //the previous fc layer returns one extra element due to the bias term
        Eigen::MatrixXd output = gradient.block(0,0,input_cache.rows(),input_cache.cols()).array() * mask.array();
        return output;
    }
};

