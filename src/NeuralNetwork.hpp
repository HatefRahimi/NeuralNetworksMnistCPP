
#include "data.hpp"
#include "labels.hpp"
#include "FullyConnected.hpp"
#include "Relu.hpp"
#include "Softmax.hpp"
#include "CrossEntropyLoss.hpp"
#include <iostream>
#include <fstream>

class NeuralNetwork {
private:
    FullyConnected fc1;
    ReLUActivation relu;
    FullyConnected fc2;
    SoftmaxActivation softmax;
    CrossEntropyLoss loss_function;

    size_t num_epochs;
    double learning_rate;
    std::ofstream log_file;

public:
    NeuralNetwork(size_t input_size, size_t hidden_size, size_t output_size, size_t epochs, double lr, const std::string& log_filepath)
        : fc1(input_size, hidden_size),
          fc2(hidden_size, output_size),
          num_epochs(epochs),
          learning_rate(lr) {
              log_file.open(log_filepath);
              if (!log_file.is_open()) {
                  std::cerr << "Failed to open log file: " << log_filepath << std::endl;
                  exit(EXIT_FAILURE);
              }
          }

    ~NeuralNetwork() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }

    void train(DataSetImages& train_data, DataSetLabels& train_labels) {
        for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
            // std::cout << "Epoch " << epoch + 1 << " / " << num_epochs << std::endl;
            for (size_t batch = 0; batch < train_data.getBatchSize(); ++batch) {
                Eigen::MatrixXd input = train_data.getBatch(batch);
                Eigen::MatrixXd labels = train_labels.getBatch(batch);

                // Forward pass
                Eigen::MatrixXd fc1Forward = fc1.forward(input);
                Eigen::MatrixXd reluForward = relu.forward(fc1Forward);
                Eigen::MatrixXd fc2Forward = fc2.forward(reluForward);
                Eigen::MatrixXd output = softmax.forward(fc2Forward);

                // Compute loss
                double loss = loss_function.forward(output, labels);
                // std::cout << "Batch " << batch << " Loss: " << loss << std::endl;

                // Backward pass
                Eigen::MatrixXd lossBackward = loss_function.backward(labels);
                Eigen::MatrixXd softmaxBackward = softmax.backward(lossBackward);
                Eigen::MatrixXd fc2Backward = fc2.backward(softmaxBackward);
                Eigen::MatrixXd reluBackward = relu.backward(fc2Backward);
                Eigen::MatrixXd fc1Backward = fc1.backward(reluBackward);
            }
        }

    }


    void test(DataSetImages& test_data, DataSetLabels& test_labels) {

            for (size_t batch = 0; batch < test_data.getBatchSize(); ++batch) {
                Eigen::MatrixXd input = test_data.getBatch(batch);
                Eigen::MatrixXd labels = test_labels.getBatch(batch);

                // Forward pass
                Eigen::MatrixXd fc1Forward = fc1.forward(input);
                Eigen::MatrixXd reluForward = relu.forward(fc1Forward);
                Eigen::MatrixXd fc2Forward = fc2.forward(reluForward);
                Eigen::MatrixXd output = softmax.forward(fc2Forward);

                // Log predictions and labels
                log_file << "Current batch: " << batch << std::endl;
                for (int i = 0; i < output.rows(); ++i) {
                    // int predicted_label = (int)Eigen::VectorXd::Map(output.row(i).data(), output.cols()).maxCoeff();
                    Eigen::Index predictedLabel;
                    Eigen::Index actualLabel;
                    test_labels.getBatch(batch).row(i).maxCoeff(&actualLabel);
                    output.row(i).maxCoeff(&predictedLabel);

                    log_file << "- image " << i << ": Prediction=" << predictedLabel
                             << " Label=" << actualLabel << std::endl;
                }

                //Eigen::Index actualLabel;
                // testLabels.getBatch(j).row(i).maxCoeff(&actualLabel);
        }

    }
};

