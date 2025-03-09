#include <iostream>
#include "NeuralNetwork.hpp"

int main(int argc, char **argv) {
    // Define hyperparameters
    size_t input_size = 784; // MNIST images are 28x28
    size_t hidden_size = stoi(argv[4]); // Number of neurons in the hidden layer
    size_t output_size = 10;  // 10 classes for digits 0-9
    size_t num_epochs = stoi(argv[2]);    // Number of training epochs
    double learning_rate = stod(argv[1]); // Learning rate for weight updates
    size_t batch_size = stoi(argv[3]);    // Size of each training batch

    // Define file paths
    std::string train_images_path = argv[5];
    std::string train_labels_path =argv[6];
    std::string log_file_path = argv[9];

    std::string test_images_path = argv[7];
    std::string test_labels_path =argv[8];

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
