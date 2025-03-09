
#include "Eigen/Dense"
#include <cmath>

class SoftmaxActivation {
private:
    Eigen::MatrixXd output_cache;
    Eigen::MatrixXd input_cache;

public:
    SoftmaxActivation() {}
    ~SoftmaxActivation() {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd &);
    Eigen::MatrixXd backward(const Eigen::MatrixXd &);
};




    Eigen::MatrixXd SoftmaxActivation::forward(const Eigen::MatrixXd& input) {
        input_cache = input;
        Eigen::MatrixXd exp_values = (input.colwise() - input.rowwise().maxCoeff());
        Eigen::MatrixXd probabilities = exp_values.array().exp();
        output_cache = probabilities.array().colwise() /probabilities.array().rowwise().sum();
        return probabilities;
    }

    Eigen::MatrixXd SoftmaxActivation::backward(const Eigen::MatrixXd& gradient) {
        // return gradient; // Gradient will be handled by the CrossEntropy loss

        Eigen::MatrixXd weightedErrorSum = (gradient.array() * output_cache.array()).rowwise().sum();

        // Broadcast the sum back to the original shape to subtract it from each errorTensor element
        Eigen::MatrixXd adjustedError = gradient.array() - (weightedErrorSum.replicate(1, gradient.cols())).array();

        // Perform the final element-wise multiplication with yHat
        Eigen::MatrixXd gradInput = output_cache.array() * adjustedError.array();

        return gradInput;
    }

