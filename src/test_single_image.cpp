#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/dense>

using namespace Eigen;
using namespace std;


int rows = 28;
int cols = 28;

// Function to read a single MNIST image from a binary file
MatrixXd readMNISTImage(const string& filename) {


    ifstream file;
    file.open(filename, std::ios::binary);

    if (!file.is_open()) {
        cerr << "Error: Could not open image file: " << filename << std::endl;
        exit(1);
    }

    file.ignore(16); // Skip the 16-byte MNIST header

    MatrixXd image(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            image(i, j) = pixel / 255.0;
        }
    }

    file.close();
    return image;
}

// Function to print the MNIST image using 1s and 0s
void printBinaryMNISTImage(const MatrixXd image) {
    cout << "\nMNIST Image Representation (Binary 1s and 0s):\n";
    for (int i = 0; i < rows ; i++) {
        for (int j = 0; j < cols ;j ++) {
            if (image(i, j) > 0.5)
                cout << "1";
            else
                cout << "0";
        }
            cout << std::endl; // New line after each row
    }
}

int main() {
    string image_file = R"(C:\Users\hatef\NeuralNetworksCPP\mnist-datasets\single-image.idx3-ubyte)";

    MatrixXd image = readMNISTImage(image_file);
    printBinaryMNISTImage(image);

    return 0;
}
