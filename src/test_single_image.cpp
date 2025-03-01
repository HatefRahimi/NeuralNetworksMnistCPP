#include <iostream>
#include <fstream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

MatrixXd readMNISTImage(const std::string& filename, int rows = 28, int cols = 28) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open image file: " << filename << std::endl;
        exit(1);
    }

    file.ignore(16); // Skip the 16-byte MNIST header

    MatrixXd image(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            image(i, j) = static_cast<double>(pixel) / 255.0; // Normalize to [0,1]
        }
    }

    file.close();
    return image;
}

void printBinaryMNISTImage(const MatrixXd& image) {
    std::cout << "\nMNIST Image Representation (Binary 1s and 0s):\n";
    for (int i = 0; i < image.rows(); i++) {
        for (int j = 0; j < image.cols(); j++) {
            std::cout << (image(i, j) > 0.5 ? "1" : "0");
        }
        std::cout << std::endl; // New line after each row
    }
}

int main() {
    std::string image_file = R"(C:\Users\hatef\TestGround\single-image.idx3-ubyte)";

    MatrixXd image = readMNISTImage(image_file);
    printBinaryMNISTImage(image);

    return 0;
}
