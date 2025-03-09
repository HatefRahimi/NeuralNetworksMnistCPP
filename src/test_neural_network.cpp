#include <iostream>
#include "NeuralNetwork.hpp"

int main() {
    // Define hyperparameters
    size_t input_size = 784; // MNIST images are 28x28
    size_t hidden_size = 500; // Number of neurons in the hidden layer
    size_t output_size = 10;  // 10 classes for digits 0-9
    size_t num_epochs = 10000;    // Number of training epochs
    double learning_rate = .001; // Learning rate for weight updates
    size_t batch_size = 1;    // Size of each training batch

    // Define file paths
    std::string train_images_path = "C:\\Users\\hatef\\NeuralNetworksCPP\\mnist-datasets\\single-image.idx3-ubyte";
    std::string train_labels_path = "C:\\Users\\hatef\\NeuralNetworksCPP\\mnist-datasets\\single-label.idx1-ubyte";
    std::string log_file_path = "log_predictions.txt";

    // Load datasets
    DataSetImages train_images(batch_size);
    train_images.readImageData(train_images_path);

    DataSetLabels train_labels(batch_size);
    train_labels.readLabelData(train_labels_path);

    // Initialize and train the neural network
    NeuralNetwork nn(input_size, hidden_size, output_size, num_epochs, learning_rate, log_file_path);
    nn.train(train_images, train_labels);

    std::cout << "Training completed. Predictions logged to " << log_file_path << std::endl;
    return 0;
}
